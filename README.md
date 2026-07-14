# Malicious Subsequence Association

This repository contains a practical implementation of **Malicious Subsequence Association (MSA)**, a prompt analysis method for studying which subsequences in an LLM prompt are most strongly associated with a model behavior.

The project is motivated by the idea that model behavior can often be traced to a relatively small part of the prompt. Here, the behavior of interest is not factual hallucination, but instead whether the model gives a **normal/compliant answer** or a **refusal**.

## What This Project Is For

The code currently supports three research cases:

- **Case 1: Benign prompts**
  The model should usually give a normal helpful response.
- **Case 2: Simple harmful prompts**
  The model should usually refuse, and short malicious subsequences may remain strongly associated with refusal even when surrounding context changes.
- **Case 3: Complex jailbreak prompts**
  The model may be induced to comply more often, and the top-scoring subsequence may be more deceptive or persuasive than overtly malicious.

## Core Idea

For a given prompt, the code:

1. gets the model's base response,
2. labels it as either `success` or `refusal`,
3. searches over candidate prompt subsequences,
4. perturbs the surrounding context while preserving one subsequence,
5. re-queries the model on those perturbed prompts,
6. measures how strongly that preserved subsequence is associated with reproducing the original behavior.

If a subsequence keeps the behavior stable across many altered contexts, that is evidence that the subsequence is important.

## MSA Score

For each prompt and candidate subsequence, the score is:

```text
MSA = log(P(target_label | subsequence) / P(target_label))
```

where:

- `target_label` is the base label of the original prompt, either `success` or `refusal`
- `P(target_label | subsequence)` is estimated from perturbations that preserve that subsequence
- `P(target_label)` is the overall prompt-level baseline estimated from all candidate perturbations in that run

Interpretation:

- `MSA > 0`: the subsequence makes the target behavior more reproducible than baseline
- `MSA = 0`: the subsequence is not especially informative relative to baseline
- `MSA < 0`: the subsequence suppresses the target behavior relative to baseline

## How The Search Works

The current implementation uses **beam search** over subsequences.

At each step:

- keep the top `B` subsequences from the previous step
- expand each one into `k` new candidates
- score those candidates
- keep only the new top `B`
- continue until the maximum subsequence length `n` is reached

The search supports both:

- contiguous subsequences
- non-contiguous subsequences

This is useful because important prompt features are not always neatly adjacent.

## Main Hyperparameters

The most important runtime parameters are:

- `B` / `--beam-width`
  Number of top subsequences kept at each beam-search step.
- `k` / `--branching-factor`
  Number of candidate expansions considered per beam element.
- `n` / `--max-subseq-words`
  Maximum subsequence length explored.
- `r` / `--repeat-runs-per-prompt`
  Number of independent MSA runs per prompt. The script keeps the strongest repeat per prompt, while also saving all repeat-level logs.
- `--variants-per-candidate`
  Number of perturbations sampled for each candidate subsequence.
- `--mask-probability`
  Probability of masking eligible surrounding words during a perturbation.
- `--model`
  Main OpenAI model used for answering prompts.
- `--refill-model`
  OpenAI model used to refill masked context.
- `--judge-model`
  OpenAI model used to label outputs as `success` or `refusal`.

## Beam Search Intuition

If `B = 3`, `k = 5`, and `n = 4`:

- step 1 evaluates single-word subsequences
- keep the best 3
- each of those 3 is expanded into 5 longer candidates
- score the 15 candidates and keep the best 3
- continue until length 4 is reached

This keeps the search tractable while still exploring a meaningful portion of the space.

## Prompt Files Included

Example prompt sets are included in `prompts/`:

- `preliminary_prompts.csv`
  Contains benign and simple harmful prompts for Cases 1 and 2.
- `case3_dataset.csv`
  Contains stronger jailbreak-style prompts for Case 3.

## Installation

Create an environment and install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Then create a `.env` file:

```bash
cp .env.example .env
```

and set:

```bash
OPENAI_API_KEY=your_key_here
```

## Running The Experiments

### Cases 1 and 2

By default, the experiment runner reads `prompts/preliminary_prompts.csv`.

```bash
python3 preliminary_experiment.py
```

A smaller smoke test:

```bash
python3 preliminary_experiment.py \
  --beam-width 2 \
  --branching-factor 3 \
  --max-subseq-words 4 \
  --variants-per-candidate 1
```

### Case 3

To run the jailbreak dataset:

```bash
python3 preliminary_experiment.py \
  --prompt-csv prompts/case3_dataset.csv \
  --output-dir results/case3_experiment \
  --beam-width 2 \
  --branching-factor 3 \
  --max-subseq-words 4 \
  --repeat-runs-per-prompt 3 \
  --variants-per-candidate 1
```

### Resume Interrupted Runs

Long runs can be resumed from the same output directory:

```bash
python3 preliminary_experiment.py --resume
```

or for a specific run:

```bash
python3 preliminary_experiment.py \
  --prompt-csv prompts/case3_dataset.csv \
  --output-dir results/case3_experiment \
  --beam-width 2 \
  --branching-factor 3 \
  --max-subseq-words 4 \
  --repeat-runs-per-prompt 3 \
  --variants-per-candidate 1 \
  --resume
```

### Dry Run Mode

To sanity-check the pipeline without API calls:

```bash
python3 preliminary_experiment.py --dry-run
```

## Plotting

After a run finishes, generate figures with:

```bash
python3 plot_preliminary_results.py --results-dir results/preliminary_experiment --output-dir results/preliminary_experiment/figures
```

For the Case 3 results, for example:

```bash
python3 plot_preliminary_results.py \
  --results-dir results/case3_experiment \
  --output-dir results/case3_experiment/figures
```

The plotting script can generate:

- top subsequence MSA score plots
- prompt-level highlighted subsequence views
- per-length highlighted views
- boxplots of top MSA score distributions
- combined comparison boxplots across result folders

## Output Files

Each run writes CSV summaries under its output directory.

The most important outputs are:

- `variant_runs.csv`
  One row per perturbed prompt / model output.
- `subsequence_summary.csv`
  Aggregate results for each candidate subsequence.
- `prompt_summary.csv`
  The top subsequence selected for each prompt.
- `repeat_variant_runs.csv`
  All repeat-level perturbation rows.
- `repeat_subsequence_summary.csv`
  All repeat-level subsequence summaries.
- `repeat_prompt_summary.csv`
  One prompt summary per repeat.
- `metadata.json`
  Run settings and checkpoint metadata.
- `figures/`
  Generated SVG plots.

## What Results Usually Look Like

Broadly speaking, the expected pattern is:

- **Case 1**: benign prompts tend to remain associated with `success`, and no obviously dangerous subsequences should stand out.
- **Case 2**: short harmful spans often remain associated with `refusal`.
- **Case 3**: the top MSA subsequence may shift away from directly malicious wording and toward more deceptive, role-playing, coercive, or persuasive phrasing.

That last case is especially interesting if the goal is to understand why some jailbreak prompts succeed more often than others.

## Important Notes

- This is a practical research implementation, not a full reproduction of the original hallucination-paper search procedure.
- Perturbations are word-level mask-and-refill operations.
- Candidate search is beam-search based rather than exhaustive.
- The code is designed for experimentation and interpretability, not for production-time filtering.

## Files In This Repo

- `preliminary_experiment.py`: main MSA runner
- `plot_preliminary_results.py`: figure generation script
- `prompts/preliminary_prompts.csv`: example prompts for Cases 1 and 2
- `prompts/case3_dataset.csv`: example prompts for Case 3
- `.env.example`: API key template
- `requirements.txt`: Python dependencies

## Citation / Acknowledgment Context

This repository is based on a research direction inspired by the paper *Why and How LLMs Hallucinate* and extends that subsequence-association perspective into malicious prompt analysis and jailbreak behavior.
