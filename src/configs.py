import argparse
import os

from dataclasses import dataclass
from pathlib import Path

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None  # type: ignore[assignment]

@dataclass
class ExperimentConfig:
    prompt_csv: Path
    output_dir: Path
    model: str
    refill_model: str
    judge_model: str
    beam_width: int
    branching_factor: int
    max_subseq_words: int
    variants_per_candidate: int
    repeat_runs_per_prompt: int
    mask_probability: float
    seed: int
    resume: bool
    dry_run: bool
    verbose: bool

def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-csv",
        type=Path,
        default=Path("prompts/preliminary_prompts.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/preliminary_experiment"),
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--refill-model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--branching-factor", type=int, default=10)
    parser.add_argument("--max-subseq-words", type=int, default=4)
    parser.add_argument("--variants-per-candidate", type=int, default=6)
    parser.add_argument("--repeat-runs-per-prompt", type=int, default=1)
    parser.add_argument("--mask-probability", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    return ExperimentConfig(
        prompt_csv=args.prompt_csv,
        output_dir=args.output_dir,
        model=args.model,
        refill_model=args.refill_model,
        judge_model=args.judge_model,
        beam_width=args.beam_width,
        branching_factor=args.branching_factor,
        max_subseq_words=args.max_subseq_words,
        variants_per_candidate=args.variants_per_candidate,
        repeat_runs_per_prompt=args.repeat_runs_per_prompt,
        mask_probability=args.mask_probability,
        seed=args.seed,
        resume=args.resume,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

def ensure_api_key(config: ExperimentConfig) -> None:
    if config.dry_run:
        return
    if OpenAI is None:
        raise SystemExit(
            "The openai package is not installed. Install it with: python3 -m pip install openai"
        )
    if os.getenv("OPENAI_API_KEY"):
        return
    raise SystemExit(
        "OPENAI_API_KEY is missing. Add it to the environment or a .env file, "
        "or rerun with --dry-run."
    )