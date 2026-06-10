# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = (
    "val/reward_score",
    "val/accuracy_reward",
    "val/format_reward",
    "reward/overall",
    "actor/ppo_kl",
    "actor/pg_loss",
    "critic/rewards/mean",
    "response_length/mean",
)


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def load_history(path: Path) -> list[dict[str, Any]]:
    history = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                history.append(flatten(json.loads(line)))
    return history


def load_generations(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    samples = []
    current: dict[str, str] = {}
    field = None
    for line in path.read_text(encoding="utf-8").splitlines():
        matched = False
        for key in ("step", "prompt", "output", "ground_truth", "score"):
            marker = f"[{key}] "
            if line.startswith(marker):
                if key == "step" and current:
                    samples.append(current)
                    current = {}
                current[key] = line[len(marker) :]
                field = key
                matched = True
                break
        if not matched and field is not None:
            current[field] = f"{current[field]}\n{line}"
    if current:
        samples.append(current)
    return [sample for sample in samples if "step" in sample]


def polyline(points: list[tuple[int, float]], width: int = 680, height: int = 220) -> str:
    if not points:
        return ""
    values = [value for _, value in points]
    low, high = min(values), max(values)
    span = high - low or 1.0
    x_span = max(len(points) - 1, 1)
    coords = []
    for index, (_, value) in enumerate(points):
        x = 12 + index * (width - 24) / x_span
        y = 10 + (high - value) * (height - 28) / span
        coords.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<polyline points="{" ".join(coords)}" fill="none" stroke="#2563eb" stroke-width="3"/>'
        f'<text x="12" y="{height - 4}" class="axis">min {low:.4g} / max {high:.4g}</text></svg>'
    )


def render_metric(history: list[dict[str, Any]], metric: str) -> str:
    points = []
    for row in history:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            points.append((int(row["step"]), float(value)))
    if not points:
        return ""
    first_step, first = points[0]
    last_step, last = points[-1]
    delta = last - first
    return (
        f"<section><h2>{html.escape(metric)}</h2>{polyline(points)}"
        f"<p>step {first_step}: <b>{first:.6g}</b> &rarr; step {last_step}: "
        f"<b>{last:.6g}</b> (delta {delta:+.6g})</p></section>"
    )


def render_generations(samples: list[dict[str, str]]) -> str:
    if not samples:
        return "<section><h2>Generations</h2><p>No step-tagged generation records found.</p></section>"
    steps = sorted({int(sample["step"]) for sample in samples})
    before_step, after_step = steps[0], steps[-1]
    before = [sample for sample in samples if int(sample["step"]) == before_step]
    after = [sample for sample in samples if int(sample["step"]) == after_step]
    before_by_prompt = {sample.get("prompt", ""): sample for sample in before}
    rows = []
    for after_sample in after:
        prompt = after_sample.get("prompt", "")
        before_sample = before_by_prompt.get(prompt, {})
        rows.append(
            "<tr>"
            f"<td><pre>{html.escape(prompt)}</pre></td>"
            f"<td><pre>{html.escape(before_sample.get('output', 'Not logged at step 0'))}</pre>"
            f"<p>score: {html.escape(before_sample.get('score', '-'))}</p></td>"
            f"<td><pre>{html.escape(after_sample.get('output', ''))}</pre>"
            f"<p>score: {html.escape(after_sample.get('score', '-'))}</p></td>"
            "</tr>"
        )
    return (
        f"<section><h2>Generation comparison: step {before_step} vs {after_step}</h2>"
        "<table><thead><tr><th>Prompt</th><th>Before</th><th>After</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML report from an EasyR1 file logger directory.")
    parser.add_argument("run_dir", type=Path, help="For example checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    args = parser.parse_args()

    log_path = args.run_dir / "experiment_log.jsonl"
    if not log_path.exists():
        raise SystemExit(f"Missing log file: {log_path}")
    history = load_history(log_path)
    output = args.output or args.run_dir / "training_report.html"
    metric_sections = "".join(render_metric(history, metric) for metric in args.metrics)
    generation_section = render_generations(load_generations(args.run_dir / "generations.log"))
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EasyR1 training report</title>
<style>
body {{ max-width: 1180px; margin: 32px auto; padding: 0 20px; font: 15px system-ui; color: #172033; }}
section {{ margin: 24px 0; padding: 18px; border: 1px solid #d8dee9; border-radius: 10px; }}
svg {{ width: 100%; height: 220px; background: #f8fafc; }} .axis {{ font-size: 12px; fill: #475569; }}
table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }} th, td {{ border: 1px solid #d8dee9; padding: 10px; vertical-align: top; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; max-height: 360px; overflow: auto; }}
</style></head><body><h1>EasyR1 GRPO training report</h1>
<p>Run directory: <code>{html.escape(str(args.run_dir))}</code></p>
{metric_sections or "<p>No selected metrics were found.</p>"}{generation_section}</body></html>"""
    output.write_text(document, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
