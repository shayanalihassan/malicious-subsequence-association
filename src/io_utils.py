import json
import time
import pandas as pd

from pathlib import Path
from typing import Dict, List, Sequence

from src.configs import ExperimentConfig



def print_summary(prompt_df: pd.DataFrame) -> None:
    if prompt_df.empty:
        print("No prompt-level results were produced.")
        return
    columns = [
        "prompt_id",
        "prompt_type",
        "base_label",
        "top_subsequence",
        "top_target_rate",
        "top_msa_score",
    ]
    print(prompt_df[columns].to_string(index=False))


def log_message(config: ExperimentConfig, message: str) -> None:
    if not config.verbose:
        return
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def normalize_prompt_records(prompt_df: pd.DataFrame) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for index, row in enumerate(prompt_df.to_dict(orient="records"), start=1):
        prompt = row.get("prompt", row.get("query", ""))
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            prompt_id = f"p{index}"
        prompt_type = row.get("prompt_type")
        if not isinstance(prompt_type, str) or not prompt_type.strip():
            prompt_type = "case3_jailbreak"
        expected_trigger = row.get("expected_trigger", "")
        if not isinstance(expected_trigger, str):
            expected_trigger = ""
        normalized = dict(row)
        normalized["prompt_id"] = prompt_id
        normalized["prompt_type"] = prompt_type
        normalized["prompt"] = prompt
        normalized["expected_trigger"] = expected_trigger
        records.append(normalized)
    return records


def read_checkpoint_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict(orient="records")


def selected_prompt_rows(
    repeat_run_rows: Sequence[Dict[str, object]],
    repeat_summary_rows: Sequence[Dict[str, object]],
    repeat_prompt_rows: Sequence[Dict[str, object]],
    repeat_runs_per_prompt: int,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    grouped_runs: Dict[str, List[Dict[str, object]]] = {}
    grouped_summaries: Dict[str, List[Dict[str, object]]] = {}
    grouped_prompts: Dict[str, List[Dict[str, object]]] = {}

    for row in repeat_run_rows:
        grouped_runs.setdefault(str(row["prompt_id"]), []).append(row)
    for row in repeat_summary_rows:
        grouped_summaries.setdefault(str(row["prompt_id"]), []).append(row)
    for row in repeat_prompt_rows:
        grouped_prompts.setdefault(str(row["prompt_id"]), []).append(row)

    selected_runs: List[Dict[str, object]] = []
    selected_summaries: List[Dict[str, object]] = []
    selected_prompts: List[Dict[str, object]] = []

    for prompt_id in sorted(grouped_prompts):
        prompt_repeat_prompt_rows = grouped_prompts[prompt_id]
        success_repeat_rows = [
            prompt_row
            for prompt_row in prompt_repeat_prompt_rows
            if prompt_row["base_label"] == "success"
        ]
        candidate_prompt_rows = success_repeat_rows if success_repeat_rows else prompt_repeat_prompt_rows
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

        selected_runs.extend(
            row
            for row in grouped_runs.get(prompt_id, [])
            if int(row["repeat_index"]) == best_repeat_index
        )
        selected_summaries.extend(
            row
            for row in grouped_summaries.get(prompt_id, [])
            if int(row["repeat_index"]) == best_repeat_index
        )
        selected_prompts.append(
            {
                **best_prompt_row,
                "selected_from_success_repeat": bool(success_repeat_rows),
                "num_repeats_considered": repeat_runs_per_prompt,
            }
        )

    return selected_runs, selected_summaries, selected_prompts


def write_outputs(
    *,
    config: ExperimentConfig,
    repeat_run_rows: Sequence[Dict[str, object]],
    repeat_summary_rows: Sequence[Dict[str, object]],
    repeat_prompt_rows: Sequence[Dict[str, object]],
    run_started: float,
    total_prompts: int,
    completed_pairs: Sequence[tuple[str, int]],
    final_write: bool,
) -> None:
    runs_df = pd.DataFrame(repeat_run_rows)
    summary_df = pd.DataFrame(repeat_summary_rows)
    prompt_repeat_df = pd.DataFrame(repeat_prompt_rows)
    selected_runs, selected_summaries, selected_prompts = selected_prompt_rows(
        repeat_run_rows=repeat_run_rows,
        repeat_summary_rows=repeat_summary_rows,
        repeat_prompt_rows=repeat_prompt_rows,
        repeat_runs_per_prompt=config.repeat_runs_per_prompt,
    )
    selected_runs_df = pd.DataFrame(selected_runs)
    selected_summary_df = pd.DataFrame(selected_summaries)
    selected_prompt_df = pd.DataFrame(selected_prompts)

    selected_runs_df.to_csv(config.output_dir / "variant_runs.csv", index=False)
    selected_summary_df.sort_values(["prompt_id", "msa_score"], ascending=[True, False]).to_csv(
        config.output_dir / "subsequence_summary.csv", index=False
    )
    selected_prompt_df.to_csv(config.output_dir / "prompt_summary.csv", index=False)
    runs_df.to_csv(config.output_dir / "repeat_variant_runs.csv", index=False)
    summary_df.sort_values(
        ["prompt_id", "repeat_index", "msa_score"], ascending=[True, True, False]
    ).to_csv(config.output_dir / "repeat_subsequence_summary.csv", index=False)
    prompt_repeat_df.to_csv(config.output_dir / "repeat_prompt_summary.csv", index=False)

    metadata = {
        "dry_run": config.dry_run,
        "model": config.model,
        "refill_model": config.refill_model,
        "judge_model": config.judge_model,
        "beam_width": config.beam_width,
        "branching_factor": config.branching_factor,
        "variants_per_candidate": config.variants_per_candidate,
        "repeat_runs_per_prompt": config.repeat_runs_per_prompt,
        "mask_probability": config.mask_probability,
        "max_subseq_words": config.max_subseq_words,
        "seed": config.seed,
        "resume": config.resume,
        "best_repeat_selection": "highest top_msa_score among success repeats if any, else highest overall",
        "checkpoint_status": "complete" if final_write else "in_progress",
        "elapsed_seconds": round(time.time() - run_started, 3),
        "total_prompts": total_prompts,
        "completed_repeat_runs": len(completed_pairs),
        "completed_repeat_keys": [
            {"prompt_id": prompt_id, "repeat_index": repeat_index}
            for prompt_id, repeat_index in sorted(completed_pairs)
        ],
    }
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_checkpoint_state(
    config: ExperimentConfig,
    prompt_records: Sequence[Dict[str, object]],
) -> tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    set[tuple[str, int]],
]:
    if not config.resume:
        return [], [], [], set()

    repeat_run_rows = read_checkpoint_csv(config.output_dir / "repeat_variant_runs.csv")
    repeat_summary_rows = read_checkpoint_csv(config.output_dir / "repeat_subsequence_summary.csv")
    repeat_prompt_rows = read_checkpoint_csv(config.output_dir / "repeat_prompt_summary.csv")
    completed_pairs = {
        (str(row["prompt_id"]), int(row["repeat_index"])) for row in repeat_prompt_rows
    }
    if not completed_pairs:
        return [], [], [], set()

    known_prompt_ids = {str(row["prompt_id"]) for row in prompt_records}
    repeat_run_rows = [
        row
        for row in repeat_run_rows
        if (str(row.get("prompt_id")), int(row.get("repeat_index", -1))) in completed_pairs
        and str(row.get("prompt_id")) in known_prompt_ids
    ]
    repeat_summary_rows = [
        row
        for row in repeat_summary_rows
        if (str(row.get("prompt_id")), int(row.get("repeat_index", -1))) in completed_pairs
        and str(row.get("prompt_id")) in known_prompt_ids
    ]
    repeat_prompt_rows = [
        row for row in repeat_prompt_rows if str(row.get("prompt_id")) in known_prompt_ids
    ]
    return repeat_run_rows, repeat_summary_rows, repeat_prompt_rows, completed_pairs

