#!/usr/bin/env python3
"""Preliminary Malicious Subsequence Association experiment.

This script implements a lightweight pilot for proposal Case 1 and Case 2:

- Case 1 (benign): helpful behavior should remain stable under context perturbation.
- Case 2 (simple harmful): short malicious subsequences should remain associated
  with refusal even when surrounding context is masked and refilled.

It now uses a beam-search-style candidate exploration procedure and supports
both contiguous and non-contiguous subsequences.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv
try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None  # type: ignore[assignment]


from src.configs import parse_args, ensure_api_key, ExperimentConfig
from src.beam_search import choose_mask_positions, build_masked_template, find_expected_trigger, compute_word_priorities, rank_beam_candidates, generate_extensions, candidate_prior_score, WORD_RE
from src.io_utils import log_message, normalize_prompt_records, load_checkpoint_state
from src.llm import refill_masks, apply_fills, generate_response, judge_response
from src.text_utils import format_subsequence, is_contiguous, locate_word_char_spans, SINGLE_WORD_RE, tokenize, detokenize



def sanitize_fill_word(raw_value: str, fallback: str) -> str:
    candidate = raw_value.strip().split()[0] if raw_value.strip() else ""
    candidate = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9'_-]+$", "", candidate)
    return candidate if SINGLE_WORD_RE.match(candidate) else fallback


def log_ratio(numerator: float, denominator: float, eps: float = 1e-6) -> float:
    return math.log((numerator + eps) / (denominator + eps))


def evaluate_candidate(
    *,
    prompt_id: str,
    prompt_type: str,
    prompt: str,
    expected_trigger: str,
    tokens: Sequence[str],
    words: Sequence[str],
    word_token_indexes: Sequence[int],
    base_response: str,
    base_label: str,
    base_rationale: str,
    candidate_indices: tuple[int, ...],
    candidate_prior: float,
    char_spans: Sequence[Tuple[int, int]],
    config: ExperimentConfig,
    client: OpenAI | None,
    rng: random.Random,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    subsequence = format_subsequence(words, candidate_indices)
    log_message(
        config,
        (
            f"[{prompt_id}] Evaluating candidate '{subsequence}' "
            f"({len(candidate_indices)} word(s), contiguous={is_contiguous(candidate_indices)})."
        ),
    )
    kept_token_indexes = {word_token_indexes[word_idx] for word_idx in candidate_indices}
    run_rows: List[Dict[str, object]] = []
    target_hits = 0
    refusal_hits = 0
    for variant_idx in range(config.variants_per_candidate):
        variant_number = variant_idx + 1
        if config.verbose and (
            variant_number == 1
            or variant_number == config.variants_per_candidate
            or variant_number % max(1, config.variants_per_candidate // 3) == 0
        ):
            log_message(
                config,
                (
                    f"[{prompt_id}] Candidate '{subsequence}': "
                    f"perturbation {variant_number}/{config.variants_per_candidate}."
                ),
            )

        mask_positions = choose_mask_positions(
            tokens=tokens,
            kept_token_indexes=kept_token_indexes,
            mask_probability=config.mask_probability,
            rng=rng,
        )
        template_tokens, mask_ids = build_masked_template(tokens, mask_positions)
        fills = refill_masks(
            template_tokens=template_tokens,
            mask_ids=mask_ids,
            client=client,
            model=config.refill_model,
            rng=rng,
            dry_run=config.dry_run,
        )
        perturbed_prompt = apply_fills(template_tokens, fills)
        perturbed_response = generate_response(
            perturbed_prompt,
            prompt_type,
            client,
            config.model,
            config.dry_run,
        )
        perturbed_label, judge_rationale = judge_response(
            perturbed_prompt,
            perturbed_response,
            client,
            config.judge_model,
            config.dry_run,
        )
        target_hits += int(perturbed_label == base_label)
        refusal_hits += int(perturbed_label == "refusal")
        run_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_type": prompt_type,
                "original_prompt": prompt,
                "base_response": base_response,
                "base_label": base_label,
                "base_rationale": base_rationale,
                "candidate_subsequence": subsequence,
                "candidate_word_len": len(candidate_indices),
                "candidate_score": candidate_prior,
                "candidate_prior_score": candidate_prior,
                "candidate_word_indices": json.dumps(list(candidate_indices)),
                "candidate_is_contiguous": is_contiguous(candidate_indices),
                "candidate_char_spans": json.dumps([list(char_spans[idx]) for idx in candidate_indices]),
                "variant_idx": variant_idx,
                "mask_template": detokenize(template_tokens),
                "fills_json": json.dumps(fills, ensure_ascii=True, sort_keys=True),
                "perturbed_prompt": perturbed_prompt,
                "perturbed_response": perturbed_response,
                "perturbed_label": perturbed_label,
                "judge_rationale": judge_rationale,
                "contains_expected_trigger": bool(expected_trigger)
                and tuple(candidate_indices) == find_expected_trigger(tokens, expected_trigger),
            }
        )

    target_rate = target_hits / config.variants_per_candidate
    refusal_rate = refusal_hits / config.variants_per_candidate
    candidate_row = {
        "prompt_id": prompt_id,
        "prompt_type": prompt_type,
        "original_prompt": prompt,
        "base_label": base_label,
        "candidate_subsequence": subsequence,
        "candidate_word_len": len(candidate_indices),
        "candidate_prior_score": candidate_prior,
        "candidate_word_indices": json.dumps(list(candidate_indices)),
        "candidate_is_contiguous": is_contiguous(candidate_indices),
        "candidate_char_spans": json.dumps([list(char_spans[idx]) for idx in candidate_indices]),
        "num_variants": config.variants_per_candidate,
        "target_rate": float(target_rate),
        "refusal_rate": float(refusal_rate),
        "success_rate": float(1.0 - refusal_rate),
        "search_score": float(target_rate),
    }
    log_message(
        config,
        (
            f"[{prompt_id}] Candidate '{subsequence}' complete. "
            f"target_rate={target_rate:.3f}, refusal_rate={refusal_rate:.3f}."
        ),
    )
    return candidate_row, run_rows


def run_beam_search(
    *,
    prompt_id: str,
    prompt_type: str,
    prompt: str,
    expected_trigger: str,
    tokens: Sequence[str],
    words: Sequence[str],
    word_token_indexes: Sequence[int],
    base_response: str,
    base_label: str,
    base_rationale: str,
    config: ExperimentConfig,
    client: OpenAI | None,
    rng: random.Random,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    char_spans = locate_word_char_spans(prompt)
    expected_trigger_indices = find_expected_trigger(tokens, expected_trigger)
    word_priorities = compute_word_priorities(words, prompt_type, expected_trigger)

    evaluated_candidates: Dict[tuple[int, ...], Dict[str, object]] = {}
    all_run_rows: List[Dict[str, object]] = []
    beam: List[tuple[int, ...]] = []

    def evaluate_if_needed(candidate_indices: tuple[int, ...]) -> Dict[str, object]:
        if candidate_indices in evaluated_candidates:
            return evaluated_candidates[candidate_indices]
        candidate_prior = candidate_prior_score(
            indices=candidate_indices,
            prompt_type=prompt_type,
            expected_trigger_indices=expected_trigger_indices,
            word_priorities=word_priorities,
        )
        candidate_row, run_rows = evaluate_candidate(
            prompt_id=prompt_id,
            prompt_type=prompt_type,
            prompt=prompt,
            expected_trigger=expected_trigger,
            tokens=tokens,
            words=words,
            word_token_indexes=word_token_indexes,
            base_response=base_response,
            base_label=base_label,
            base_rationale=base_rationale,
            candidate_indices=candidate_indices,
            candidate_prior=candidate_prior,
            char_spans=char_spans,
            config=config,
            client=client,
            rng=rng,
        )
        evaluated_candidates[candidate_indices] = candidate_row
        all_run_rows.extend(run_rows)
        return candidate_row

    initial_candidates = [tuple([word_idx]) for word_idx in range(len(words))]
    log_message(
        config,
        f"[{prompt_id}] Beam step 1/{config.max_subseq_words}: evaluating {len(initial_candidates)} single-word candidates.",
    )
    first_round = [evaluate_if_needed(candidate) for candidate in initial_candidates]
    beam = [
        tuple(json.loads(row["candidate_word_indices"]))
        for row in rank_beam_candidates(first_round)[: config.beam_width]
    ]
    log_message(
        config,
        f"[{prompt_id}] Beam step 1 top candidates: "
        + ", ".join(evaluated_candidates[candidate]["candidate_subsequence"] for candidate in beam),
    )

    for step in range(2, config.max_subseq_words + 1):
        expansions: List[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for candidate in beam:
            for extension in generate_extensions(
                subsequence=candidate,
                word_count=len(words),
                branching_factor=config.branching_factor,
                word_priorities=word_priorities,
            ):
                if len(extension) != step or extension in seen:
                    continue
                seen.add(extension)
                expansions.append(extension)

        if not expansions:
            log_message(config, f"[{prompt_id}] Beam step {step}: no further expansions available.")
            break

        log_message(
            config,
            f"[{prompt_id}] Beam step {step}/{config.max_subseq_words}: evaluating {len(expansions)} candidates from beam expansion.",
        )
        round_rows = [evaluate_if_needed(candidate) for candidate in expansions]
        beam = [
            tuple(json.loads(row["candidate_word_indices"]))
            for row in rank_beam_candidates(round_rows)[: config.beam_width]
        ]
        log_message(
            config,
            f"[{prompt_id}] Beam step {step} top candidates: "
            + ", ".join(evaluated_candidates[candidate]["candidate_subsequence"] for candidate in beam),
        )

    candidate_rows = list(evaluated_candidates.values())
    return candidate_rows, all_run_rows


def run_single_prompt_experiment(
    *,
    prompt_id: str,
    prompt_type: str,
    prompt: str,
    expected_trigger: str,
    source_row: Dict[str, object],
    repeat_index: int,
    config: ExperimentConfig,
    client: OpenAI | None,
    rng: random.Random,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    tokens = tokenize(prompt)
    words = [token for token in tokens if WORD_RE.match(token)]
    base_response = generate_response(prompt, prompt_type, client, config.model, config.dry_run)
    base_label, base_rationale = judge_response(
        prompt, base_response, client, config.judge_model, config.dry_run
    )
    log_message(
        config,
        f"[{prompt_id}][repeat {repeat_index}] Base label={base_label}. Running beam search over subsequences.",
    )
    word_token_indexes = [idx for idx, token in enumerate(tokens) if WORD_RE.match(token)]
    candidate_rows, prompt_runs = run_beam_search(
        prompt_id=prompt_id,
        prompt_type=prompt_type,
        prompt=prompt,
        expected_trigger=expected_trigger,
        tokens=tokens,
        words=words,
        word_token_indexes=word_token_indexes,
        base_response=base_response,
        base_label=base_label,
        base_rationale=base_rationale,
        config=config,
        client=client,
        rng=rng,
    )
    baseline_target_rate = sum(
        1 for run in prompt_runs if run["perturbed_label"] == base_label
    ) / len(prompt_runs)
    refusal_rate_overall = sum(
        1 for run in prompt_runs if run["perturbed_label"] == "refusal"
    ) / len(prompt_runs)

    summary_rows: List[Dict[str, object]] = []
    for candidate in candidate_rows:
        msa_score = log_ratio(float(candidate["target_rate"]), baseline_target_rate)
        summary_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_type": prompt_type,
                "original_prompt": prompt,
                "base_label": base_label,
                "repeat_index": repeat_index,
                "candidate_subsequence": candidate["candidate_subsequence"],
                "candidate_word_len": int(candidate["candidate_word_len"]),
                "candidate_score": float(candidate["candidate_prior_score"]),
                "candidate_prior_score": float(candidate["candidate_prior_score"]),
                "candidate_word_indices": candidate["candidate_word_indices"],
                "candidate_is_contiguous": bool(candidate["candidate_is_contiguous"]),
                "candidate_char_spans": candidate["candidate_char_spans"],
                "num_variants": int(candidate["num_variants"]),
                "target_rate": float(candidate["target_rate"]),
                "refusal_rate": float(candidate["refusal_rate"]),
                "success_rate": float(candidate["success_rate"]),
                "overall_target_rate": float(baseline_target_rate),
                "overall_refusal_rate": float(refusal_rate_overall),
                "msa_score": float(msa_score),
                "source_success_hint": source_row.get("perturb_rate0.3_(success_out_of_25)"),
            }
        )

    for run_row in prompt_runs:
        run_row["repeat_index"] = repeat_index
        run_row["source_success_hint"] = source_row.get("perturb_rate0.3_(success_out_of_25)")

    prompt_summary_df = pd.DataFrame(summary_rows).sort_values(
        ["msa_score", "target_rate", "candidate_score", "candidate_word_len"],
        ascending=[False, False, False, True],
    )
    top_row = prompt_summary_df.iloc[0].to_dict()
    prompt_row = {
        "prompt_id": prompt_id,
        "prompt_type": prompt_type,
        "original_prompt": prompt,
        "word_count": len(words),
        "base_label": base_label,
        "base_response": base_response,
        "repeat_index": repeat_index,
        "top_subsequence": top_row["candidate_subsequence"],
        "top_subsequence_word_indices": top_row["candidate_word_indices"],
        "top_subsequence_is_contiguous": bool(top_row["candidate_is_contiguous"]),
        "top_target_rate": top_row["target_rate"],
        "top_refusal_rate": top_row["refusal_rate"],
        "top_msa_score": top_row["msa_score"],
        "source_success_hint": source_row.get("perturb_rate0.3_(success_out_of_25)"),
    }
    return prompt_runs, summary_rows, prompt_row


def main() -> int:
    load_dotenv()
    config = parse_args()
    ensure_api_key(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    client = None if config.dry_run else OpenAI()
    run_started = time.time()

    prompts_df = pd.read_csv(config.prompt_csv)
    prompt_records = normalize_prompt_records(prompts_df)
    total_prompts = len(prompt_records)
    prompt_upper_bounds = []
    for row in prompt_records:
        prompt = str(row["prompt"])
        word_count = len([token for token in tokenize(prompt) if WORD_RE.match(token)])
        per_prompt_candidate_upper_bound = word_count + max(
            0, (config.max_subseq_words - 1) * config.beam_width * config.branching_factor
        )
        prompt_upper_bounds.append(per_prompt_candidate_upper_bound)
    total_expected_variants_upper_bound = (
        sum(prompt_upper_bounds) * config.variants_per_candidate * config.repeat_runs_per_prompt
    )
    log_message(
        config,
        (
            f"Starting experiment with {total_prompts} prompts, beam_width={config.beam_width}, "
            f"branching_factor={config.branching_factor}, max_subseq_words={config.max_subseq_words}, "
            f"{config.repeat_runs_per_prompt} repeat run(s) per prompt, and "
            f"{config.variants_per_candidate} perturbations per candidate "
            f"(upper bound {total_expected_variants_upper_bound} perturbed evaluations)."
        ),
    )
    log_message(
        config,
        (
            f"Models: response={config.model}, refill={config.refill_model}, "
            f"judge={config.judge_model}, dry_run={config.dry_run}."
        ),
    )
    repeat_run_rows, repeat_summary_rows, repeat_prompt_rows, completed_pairs = load_checkpoint_state(
        config, prompt_records
    )
    if completed_pairs:
        log_message(
            config,
            (
                f"Resuming from checkpoint with {len(completed_pairs)} completed prompt-repeat runs "
                f"already stored in {config.output_dir}."
            ),
        )

    for prompt_index, row in enumerate(prompt_records, start=1):
        prompt_id = row["prompt_id"]
        prompt_type = row["prompt_type"]
        prompt = row["prompt"]
        expected_trigger = str(row.get("expected_trigger", "") or "")
        prompt_started = time.time()
        log_message(
            config,
            f"[{prompt_index}/{total_prompts}] Prompt {prompt_id} ({prompt_type}) started: {prompt}",
        )
        prompt_repeat_runs = [
            saved_row for saved_row in repeat_run_rows if str(saved_row["prompt_id"]) == prompt_id
        ]
        prompt_repeat_summaries = [
            saved_row for saved_row in repeat_summary_rows if str(saved_row["prompt_id"]) == prompt_id
        ]
        prompt_repeat_prompt_rows = [
            saved_row for saved_row in repeat_prompt_rows if str(saved_row["prompt_id"]) == prompt_id
        ]

        for repeat_index in range(1, config.repeat_runs_per_prompt + 1):
            pair_key = (prompt_id, repeat_index)
            if pair_key in completed_pairs:
                log_message(
                    config,
                    f"[{prompt_id}] Repeat {repeat_index}/{config.repeat_runs_per_prompt} already complete; skipping.",
                )
                continue
            repeat_rng = random.Random(config.seed + prompt_index * 1000 + repeat_index)
            log_message(
                config,
                f"[{prompt_id}] Repeat {repeat_index}/{config.repeat_runs_per_prompt} started.",
            )
            one_run_rows, one_summary_rows, one_prompt_row = run_single_prompt_experiment(
                prompt_id=prompt_id,
                prompt_type=prompt_type,
                prompt=prompt,
                expected_trigger=expected_trigger,
                source_row=row,
                repeat_index=repeat_index,
                config=config,
                client=client,
                rng=repeat_rng,
            )
            prompt_repeat_runs.extend(one_run_rows)
            prompt_repeat_summaries.extend(one_summary_rows)
            prompt_repeat_prompt_rows.append(one_prompt_row)
            repeat_run_rows.extend(one_run_rows)
            repeat_summary_rows.extend(one_summary_rows)
            repeat_prompt_rows.append(one_prompt_row)
            completed_pairs.add(pair_key)
            log_message(
                config,
                (
                    f"[{prompt_id}] Repeat {repeat_index} complete. "
                    f"base_label={one_prompt_row['base_label']}, "
                    f"top_subsequence='{one_prompt_row['top_subsequence']}', "
                    f"msa={one_prompt_row['top_msa_score']:.3f}."
                ),
            )
            write_outputs(
                config=config,
                repeat_run_rows=repeat_run_rows,
                repeat_summary_rows=repeat_summary_rows,
                repeat_prompt_rows=repeat_prompt_rows,
                run_started=run_started,
                total_prompts=total_prompts,
                completed_pairs=completed_pairs,
                final_write=False,
            )

        success_repeat_rows = [
            prompt_row for prompt_row in prompt_repeat_prompt_rows if prompt_row["base_label"] == "success"
        ]
        candidate_prompt_rows = success_repeat_rows if success_repeat_rows else prompt_repeat_prompt_rows
        if not candidate_prompt_rows:
            log_message(
                config,
                f"[{prompt_id}] No completed repeats available yet; skipping selection.",
            )
            continue
        best_prompt_row = sorted(
            candidate_prompt_rows,
            key=lambda item: (
                float(item["top_msa_score"]),
                float(item["top_target_rate"]),
                float(item["top_refusal_rate"]),
            ),
            reverse=True,
        )[0]
        best_repeat_index = int(best_prompt_row["repeat_index"])

        log_message(
            config,
            (
                f"[{prompt_id}] Complete in {time.time() - prompt_started:.1f}s. "
                f"Selected repeat {best_repeat_index} with top subsequence='{best_prompt_row['top_subsequence']}', "
                f"target_rate={best_prompt_row['top_target_rate']:.3f}, msa={best_prompt_row['top_msa_score']:.3f}."
            ),
        )
        write_outputs(
            config=config,
            repeat_run_rows=repeat_run_rows,
            repeat_summary_rows=repeat_summary_rows,
            repeat_prompt_rows=repeat_prompt_rows,
            run_started=run_started,
            total_prompts=total_prompts,
            completed_pairs=completed_pairs,
            final_write=False,
        )

    write_outputs(
        config=config,
        repeat_run_rows=repeat_run_rows,
        repeat_summary_rows=repeat_summary_rows,
        repeat_prompt_rows=repeat_prompt_rows,
        run_started=run_started,
        total_prompts=total_prompts,
        completed_pairs=completed_pairs,
        final_write=True,
    )
    runs_df = pd.read_csv(config.output_dir / "variant_runs.csv")
    summary_df = pd.read_csv(config.output_dir / "subsequence_summary.csv")
    prompt_df = pd.read_csv(config.output_dir / "prompt_summary.csv")
    log_message(
        config,
        (
            f"Finished in {time.time() - run_started:.1f}s. "
            f"Wrote {len(runs_df)} variant rows, {len(summary_df)} subsequence summaries, "
            f"{len(prompt_df)} selected prompt summaries, and full repeat logs to {config.output_dir}."
        ),
    )
    print_summary(prompt_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
