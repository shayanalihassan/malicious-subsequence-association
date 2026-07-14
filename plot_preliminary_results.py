#!/usr/bin/env python3
"""Create lightweight SVG plots for the preliminary MSA experiment."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/preliminary_experiment"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/preliminary_experiment/figures"),
    )
    parser.add_argument(
        "--compare-results-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #1f2937; }',
        '.title { font-size: 20px; font-weight: 700; }',
        '.subtitle { font-size: 12px; fill: #4b5563; }',
        '.axis { font-size: 12px; }',
        '.label { font-size: 12px; }',
        '.small { font-size: 11px; fill: #4b5563; }',
        '</style>',
        '<rect x="0" y="0" width="100%" height="100%" fill="#fcfcfd" />',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def esc(text: object) -> str:
    return html.escape(str(text))


def color_for_prompt_type(prompt_type: str) -> str:
    if prompt_type == "benign":
        return "#1f7a3d"
    if prompt_type == "simple_harmful":
        return "#b91c1c"
    return "#8b5cf6"


def prompt_type_label(prompt_type: str) -> str:
    if prompt_type == "benign":
        return "Benign"
    if prompt_type == "simple_harmful":
        return "Simple Harmful"
    if prompt_type == "case3_jailbreak":
        return "Complex Jailbreak"
    return prompt_type.replace("_", " ").title()


def remove_old_outputs(output_dir: Path) -> None:
    for path in output_dir.glob("prompt_top_subsequence_highlight*.svg"):
        path.unlink()
    for path in output_dir.glob("top_subsequence_msa*.pdf"):
        path.unlink()
    for path in output_dir.glob("top_msa_score_boxplot*.pdf"):
        path.unlink()
    names = [
        "top_subsequence_target_rate.svg",
        "top_subsequence_label_mix.svg",
        "subsequence_refusal_heatmap.svg",
        "top_subsequence_msa.pdf",
        "prompt_top_subsequence_highlight.pdf",
        "top_msa_score_boxplot.pdf",
    ]
    for name in names:
        path = output_dir / name
        if path.exists():
            path.unlink()


def draw_bar_chart(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    secondary_col: str,
    title: str,
    subtitle: str,
    x_axis_label: str,
    output_path: Path,
    value_min: float | None = None,
    value_max: float | None = None,
) -> None:
    row_h = 42
    left = 300
    right = 80
    top = 102
    bottom = 60
    width = 1080
    height = top + bottom + row_h * len(df)
    plot_width = width - left - right

    min_val = float(df[value_col].min() if value_min is None else value_min)
    max_val = float(df[value_col].max() if value_max is None else value_max)
    if min_val == max_val:
        if min_val == 0:
            max_val = 1.0
        else:
            min_val = min(0.0, min_val)
            max_val = max_val * 1.1

    def to_x(value: float) -> float:
        return left + (value - min_val) / (max_val - min_val) * plot_width

    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="40" y="42">{esc(title)}</text>')
    lines.append(f'<text class="subtitle" x="40" y="64">{esc(subtitle)}</text>')

    zero_x = to_x(0)
    lines.append(
        f'<line x1="{zero_x:.1f}" y1="{top - 10}" x2="{zero_x:.1f}" y2="{height - bottom + 10}" '
        'stroke="#9ca3af" stroke-width="1.5" />'
    )

    for tick_idx in range(6):
        tick_val = min_val + (max_val - min_val) * tick_idx / 5
        tick_x = to_x(tick_val)
        lines.append(
            f'<line x1="{tick_x:.1f}" y1="{top - 10}" x2="{tick_x:.1f}" y2="{height - bottom}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        lines.append(
            f'<text class="axis" x="{tick_x:.1f}" y="{height - bottom + 24}" text-anchor="middle">'
            f'{tick_val:.2f}</text>'
        )

    lines.append(
        f'<text class="axis" x="{left + plot_width / 2:.1f}" y="{height - 16}" text-anchor="middle">'
        f'{esc(x_axis_label)}</text>'
    )

    for row_idx, row in df.reset_index(drop=True).iterrows():
        y = top + row_idx * row_h
        value = float(row[value_col])
        bar_x = min(zero_x, to_x(value))
        bar_w = abs(to_x(value) - zero_x)
        color = color_for_prompt_type(str(row[secondary_col]))
        lines.append(
            f'<text class="label" x="{left - 12}" y="{y + 17}" text-anchor="end">{esc(row[label_col])}</text>'
        )
        lines.append(
            f'<text class="small" x="{left - 12}" y="{y + 32}" text-anchor="end">{esc(row[secondary_col])}</text>'
        )
        lines.append(
            f'<rect x="{bar_x:.1f}" y="{y}" width="{max(bar_w, 1):.1f}" height="22" rx="4" fill="{color}" />'
        )
        lines.append(
            f'<text class="label" x="{to_x(value) + (8 if value >= 0 else -8):.1f}" y="{y + 16}" '
            f'text-anchor="{"start" if value >= 0 else "end"}">{value:.3f}</text>'
        )

    write_svg(output_path, lines)


def build_highlight_segments(prompt: str, word_indices_json: str | float | None) -> list[tuple[str, bool]]:
    if not isinstance(word_indices_json, str) or not word_indices_json.strip():
        return [(prompt, False)]
    try:
        selected_indices = {int(value) for value in json.loads(word_indices_json)}
    except Exception:
        return [(prompt, False)]

    matches = list(re.finditer(r"\w+(?:'\w+)?", prompt))
    spans: list[tuple[int, int]] = []
    for idx, match in enumerate(matches):
        if idx in selected_indices:
            spans.append((match.start(), match.end()))
    if not spans:
        return [(prompt, False)]

    segments: list[tuple[str, bool]] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            segments.append((prompt[cursor:start], False))
        segments.append((prompt[start:end], True))
        cursor = end
    if cursor < len(prompt):
        segments.append((prompt[cursor:], False))
    return segments


def selected_char_spans(prompt: str, word_indices_json: str | float | None) -> list[tuple[int, int]]:
    if not isinstance(word_indices_json, str) or not word_indices_json.strip():
        return []
    try:
        selected_indices = {int(value) for value in json.loads(word_indices_json)}
    except Exception:
        return []

    spans: list[tuple[int, int]] = []
    for idx, match in enumerate(re.finditer(r"\w+(?:'\w+)?", prompt)):
        if idx in selected_indices:
            spans.append((match.start(), match.end()))
    return spans


def split_prompt_for_display(
    prompt: str,
    highlight_spans: list[tuple[int, int]],
    max_lines: int = 2,
) -> list[tuple[str, list[tuple[int, int]]]]:
    if max_lines <= 1 or len(prompt) < 85:
        return [(prompt, highlight_spans)]

    midpoint = len(prompt) // 2
    all_breaks = [match.start() for match in re.finditer(r"\s+", prompt)]
    if not all_breaks:
        return [(prompt, highlight_spans)]

    protected_positions: set[int] = set()
    for start, end in highlight_spans:
        protected_positions.update(range(start, end))
    valid_breaks = [pos for pos in all_breaks if pos not in protected_positions] or all_breaks
    split_at = min(valid_breaks, key=lambda pos: abs(pos - midpoint))

    left_text = prompt[:split_at].rstrip()
    right_raw = prompt[split_at:]
    right_text = right_raw.lstrip()
    right_shift = len(prompt[:split_at]) + (len(right_raw) - len(right_text))

    left_spans: list[tuple[int, int]] = []
    right_spans: list[tuple[int, int]] = []
    for start, end in highlight_spans:
        if end <= split_at:
            left_spans.append((start, end))
        elif start >= right_shift:
            right_spans.append((start - right_shift, end - right_shift))
        else:
            left_end = min(end, split_at)
            right_start = max(start, right_shift)
            if left_end > start:
                left_spans.append((start, left_end))
            if end > right_start:
                right_spans.append((right_start - right_shift, end - right_shift))
    return [(left_text, left_spans), (right_text, right_spans)]


def build_segments_from_spans(text: str, spans: list[tuple[int, int]]) -> list[tuple[str, bool]]:
    if not spans:
        return [(text, False)]
    segments: list[tuple[str, bool]] = []
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            segments.append((text[cursor:start], False))
        segments.append((text[start:end], True))
        cursor = end
    if cursor < len(text):
        segments.append((text[cursor:], False))
    return segments


def draw_prompt_highlight_view(
    prompt_df: pd.DataFrame,
    output_path: Path,
    title: str,
    *,
    max_prompt_lines: int = 1,
) -> None:
    df = prompt_df.sort_values(["prompt_type", "prompt_id"]).reset_index(drop=True)
    row_h = 78 if max_prompt_lines > 1 else 54
    left = 48
    right = 86
    top = 74
    bottom = 14
    prepared_rows: list[dict[str, object]] = []
    longest_visible_line = 80
    for _, row in df.iterrows():
        prompt = str(row["original_prompt"])
        spans = selected_char_spans(
            prompt,
            row["top_subsequence_word_indices"] if "top_subsequence_word_indices" in row else None,
        )
        prompt_lines = split_prompt_for_display(prompt, spans, max_lines=max_prompt_lines)
        longest_visible_line = max(
            longest_visible_line,
            max(len(line_text) for line_text, _ in prompt_lines),
        )
        prepared_rows.append({"row": row, "prompt_lines": prompt_lines})
    width_limit = 1450 if max_prompt_lines > 1 else 2200
    width = min(max(720, 210 + int(longest_visible_line * 7.1)), width_limit)
    height = top + bottom + row_h * len(df)
    char_w = 7.4

    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="40" y="42">{esc(title)}</text>')

    for row_idx, prepared in enumerate(prepared_rows):
        row = prepared["row"]
        prompt_lines = prepared["prompt_lines"]
        y = top + row_idx * row_h
        prompt_type = str(row["prompt_type"])
        msa_score = float(
            row["display_msa_score"] if "display_msa_score" in row else row["top_msa_score"]
        )
        accent = color_for_prompt_type(prompt_type)
        box_y = y - (30 if max_prompt_lines > 1 else 18)
        box_h = 60 if max_prompt_lines > 1 else 36

        lines.append(
            f'<rect x="24" y="{box_y}" width="{width - 48}" height="{box_h}" rx="8" fill="#ffffff" stroke="#e5e7eb" />'
        )
        lines.append(
            f'<rect x="28" y="{y - (24 if max_prompt_lines > 1 else 12)}" width="6" height="{48 if max_prompt_lines > 1 else 24}" rx="3" fill="{accent}" />'
        )

        for line_idx, (line_text, line_spans) in enumerate(prompt_lines):
            baseline_y = y - 8 + line_idx * 18 if max_prompt_lines > 1 else y + 2
            highlight_y = baseline_y - 14
            cursor_x = left
            for segment_text, is_highlighted in build_segments_from_spans(line_text, line_spans):
                segment_w = len(segment_text) * char_w
                if is_highlighted:
                    lines.append(
                        f'<rect x="{cursor_x - 2:.1f}" y="{highlight_y:.1f}" width="{max(segment_w + 4, 12):.1f}" height="18" '
                        f'rx="4" fill="{accent}" fill-opacity="0.18" stroke="{accent}" stroke-width="1" />'
                    )
                    lines.append(
                        f'<text x="{cursor_x:.1f}" y="{baseline_y:.1f}" '
                        'style="font-family: Menlo, Consolas, monospace; font-size: 12px; font-weight: 700;">'
                        f'{esc(segment_text)}</text>'
                    )
                else:
                    lines.append(
                        f'<text x="{cursor_x:.1f}" y="{baseline_y:.1f}" style="font-family: Menlo, Consolas, monospace; font-size: 12px;">'
                        f'{esc(segment_text)}</text>'
                    )
                cursor_x += segment_w
        lines.append(
            f'<text class="small" x="{width - 32}" y="{y + (8 if max_prompt_lines > 1 else 4)}" text-anchor="end">MSA {msa_score:.3f}</text>'
        )

    write_svg(output_path, lines)


def top_candidate_per_length(summary_df: pd.DataFrame, candidate_len: int) -> pd.DataFrame:
    filtered = summary_df[summary_df["candidate_word_len"] == candidate_len].copy()
    if filtered.empty:
        return filtered
    return (
        filtered.sort_values(
            ["prompt_id", "msa_score", "target_rate", "candidate_score"],
            ascending=[True, False, False, False],
        )
        .groupby("prompt_id", as_index=False)
        .head(1)
        .rename(
            columns={
                "candidate_subsequence": "top_subsequence",
                "candidate_word_indices": "top_subsequence_word_indices",
                "refusal_rate": "top_refusal_rate",
                "target_rate": "top_target_rate",
                "msa_score": "top_msa_score",
            }
        )
    )


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def draw_top_msa_boxplot(prompt_df: pd.DataFrame, output_path: Path) -> None:
    unique_prompt_types = list(dict.fromkeys(prompt_df["prompt_type"].astype(str).tolist()))
    groups = [(prompt_type_label(key), key) for key in unique_prompt_types]
    values_by_group = {
        label: sorted(prompt_df[prompt_df["prompt_type"] == key]["top_msa_score"].astype(float).tolist())
        for label, key in groups
    }
    all_values = [value for values in values_by_group.values() for value in values]
    y_min = min(0.0, min(all_values) if all_values else 0.0)
    y_max = max(all_values) if all_values else 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    width = 760
    height = 520
    left = 90
    right = 40
    top = 102
    bottom = 80
    plot_height = height - top - bottom
    plot_width = width - left - right

    def to_y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = svg_header(width, height)
    lines.append('<text class="title" x="40" y="42">Distribution of Top MSA Scores by Prompt Category</text>')

    for tick_idx in range(6):
        tick_val = y_min + (y_max - y_min) * tick_idx / 5
        y = to_y(tick_val)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1" />'
        )
        lines.append(
            f'<text class="axis" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick_val:.2f}</text>'
        )

    lines.append(
        f'<text class="axis" x="24" y="{top + plot_height / 2:.1f}" transform="rotate(-90, 24, {top + plot_height / 2:.1f})" text-anchor="middle">Top MSA score</text>'
    )

    for idx, (label, key) in enumerate(groups):
        values = values_by_group[label]
        center_x = left + plot_width * (idx + 0.5) / len(groups)
        color = color_for_prompt_type(key)
        lines.append(
            f'<text class="label" x="{center_x:.1f}" y="{height - 28}" text-anchor="middle">{esc(label)}</text>'
        )
        if not values:
            continue
        q1 = percentile(values, 0.25)
        q2 = percentile(values, 0.50)
        q3 = percentile(values, 0.75)
        vmin = min(values)
        vmax = max(values)
        box_w = 120
        x0 = center_x - box_w / 2
        lines.append(
            f'<line x1="{center_x:.1f}" y1="{to_y(vmin):.1f}" x2="{center_x:.1f}" y2="{to_y(vmax):.1f}" stroke="{color}" stroke-width="2" />'
        )
        lines.append(
            f'<rect x="{x0:.1f}" y="{to_y(q3):.1f}" width="{box_w:.1f}" height="{max(to_y(q1) - to_y(q3), 1):.1f}" rx="6" fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="2" />'
        )
        lines.append(
            f'<line x1="{x0:.1f}" y1="{to_y(q2):.1f}" x2="{x0 + box_w:.1f}" y2="{to_y(q2):.1f}" stroke="{color}" stroke-width="2.5" />'
        )
        lines.append(
            f'<line x1="{x0 + 20:.1f}" y1="{to_y(vmin):.1f}" x2="{x0 + box_w - 20:.1f}" y2="{to_y(vmin):.1f}" stroke="{color}" stroke-width="2" />'
        )
        lines.append(
            f'<line x1="{x0 + 20:.1f}" y1="{to_y(vmax):.1f}" x2="{x0 + box_w - 20:.1f}" y2="{to_y(vmax):.1f}" stroke="{color}" stroke-width="2" />'
        )
    write_svg(output_path, lines)


def draw_combined_top_msa_boxplot(
    prompt_dfs: list[pd.DataFrame],
    output_path: Path,
) -> None:
    combined_df = pd.concat(prompt_dfs, ignore_index=True)
    draw_top_msa_boxplot(combined_df, output_path)


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = results_dir / "prompt_summary.csv"
    summary_path = results_dir / "subsequence_summary.csv"
    if not prompt_path.exists():
        raise SystemExit(
            "Missing experiment outputs. Run preliminary_experiment.py first so prompt_summary.csv exists."
        )
    if not summary_path.exists():
        raise SystemExit(
            "Missing subsequence summary output. Run preliminary_experiment.py first so subsequence_summary.csv exists."
        )

    prompt_df = pd.read_csv(prompt_path)
    summary_df = pd.read_csv(summary_path)
    prompt_df["display_msa_score"] = prompt_df["top_msa_score"]
    remove_old_outputs(output_dir)

    draw_bar_chart(
        df=prompt_df.sort_values("top_msa_score", ascending=True),
        value_col="top_msa_score",
        label_col="prompt_id",
        secondary_col="prompt_type",
        title="Top Subsequence MSA Score",
        subtitle="Each bar shows the highest prompt-level MSA score found within that prompt.",
        x_axis_label="MSA score",
        output_path=output_dir / "top_subsequence_msa.svg",
    )
    draw_prompt_highlight_view(
        prompt_df=prompt_df,
        output_path=output_dir / "prompt_top_subsequence_highlight.svg",
        title="Prompt View With Top-Scoring Subsequence Highlighted",
    )
    refined_prompt_ids = {"p1", "p2", "p3", "p4", "p6", "p8", "p10"}
    refined_prompt_df = prompt_df[prompt_df["prompt_id"].astype(str).isin(refined_prompt_ids)].copy()
    if not refined_prompt_df.empty:
        draw_prompt_highlight_view(
            prompt_df=refined_prompt_df,
            output_path=output_dir / "prompt_top_subsequence_highlight_refined.svg",
            title="Selected Case 3 Prompts With Top-Scoring Subsequences Highlighted",
            max_prompt_lines=2,
        )
    draw_top_msa_boxplot(
        prompt_df=prompt_df,
        output_path=output_dir / "top_msa_score_boxplot.svg",
    )

    if args.compare_results_dir is not None:
        compare_prompt_path = args.compare_results_dir / "prompt_summary.csv"
        if not compare_prompt_path.exists():
            raise SystemExit(
                f"Missing compare prompt summary: {compare_prompt_path}"
            )
        compare_prompt_df = pd.read_csv(compare_prompt_path)
        draw_combined_top_msa_boxplot(
            prompt_dfs=[compare_prompt_df, prompt_df],
            output_path=output_dir / "combined_top_msa_score_boxplot.svg",
        )

    available_lengths = sorted(
        int(value) for value in summary_df["candidate_word_len"].dropna().unique().tolist()
    )
    for candidate_len in available_lengths:
        per_length_df = top_candidate_per_length(summary_df, candidate_len)
        if per_length_df.empty:
            continue
        per_length_df["display_msa_score"] = per_length_df["top_msa_score"]
        draw_prompt_highlight_view(
            prompt_df=per_length_df,
            output_path=output_dir / f"prompt_top_subsequence_highlight_len{candidate_len}.svg",
            title=f"Prompt View With Top-Scoring Length-{candidate_len} Subsequences Highlighted",
        )

    print(f"Wrote figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
