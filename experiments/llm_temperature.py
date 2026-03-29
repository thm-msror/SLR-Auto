from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas.inital_screen.gpt_screener_initial import (
    build_prompt,
    count_answers,
    parse_screening_answers,
    relevance_score,
)
from atlas.inital_screen.prompts import SCREEN_INITIAL_PROMPT
from atlas.utils.gpt_client import call_gpt_chat


DEFAULT_TEMPERATURES = (0.0, 0.2, 0.5, 1.0)
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_NONZERO_REPEATS = 3
DEFAULT_RANDOM_SEED = 42


@dataclass
class SampleRecord:
    record_id: str
    title: str
    abstract: str
    label_included: int
    year: str = ""
    publisher: str = ""
    source_dataset: str = ""


@dataclass
class TrialResult:
    temperature: float
    repeat_index: int
    record_id: str
    gold_label: int
    predicted_label: int
    relevance_score: int
    include_yes: int
    include_no: int
    include_insufficient: int
    exclude_yes: int
    exclude_no: int
    exclude_insufficient: int
    format_compliant: int
    raw: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quick temperature-ablation experiment for title/abstract screening "
            "using a SYNERGY-style dataset."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a local SYNERGY export in CSV, JSON, or JSONL format.",
    )
    parser.add_argument(
        "--criteria-file",
        required=True,
        help="Path to a text file containing one inclusion/exclusion criterion per line.",
    )
    parser.add_argument(
        "--topic",
        help=(
            "Optional dataset/topic name to filter on, e.g. "
            "'Hall_2012' or 'Radjenovic_2013'."
        ),
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEMPERATURES),
        help="Temperature sweep to evaluate. Default: 0.0 0.2 0.5 1.0",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of records to sample once and reuse across all runs.",
    )
    parser.add_argument(
        "--nonzero-repeats",
        type=int,
        default=DEFAULT_NONZERO_REPEATS,
        help="Number of repeats for each nonzero temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for the fixed sample.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/results",
        help="Directory for CSV/JSON outputs.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".js"}:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict):
            for key in ("records", "data", "items", "results"):
                value = raw.get(key)
                if isinstance(value, list):
                    return pd.DataFrame(value)
        raise ValueError("Unsupported JSON structure. Expected a list of objects.")
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def pick_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def load_criteria(path: Path) -> List[str]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"No criteria found in {path}")
    return lines


def normalize_label(value: Any) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    text = str(value).strip().lower()
    return 1 if text in {"1", "true", "yes", "include", "included", "relevant"} else 0


def load_synergy_records(path: Path, topic: Optional[str]) -> List[SampleRecord]:
    df = load_table(path)

    topic_col = pick_column(
        df.columns,
        ("dataset", "dataset_id", "review_name", "review", "topic", "collection"),
    )
    title_col = pick_column(df.columns, ("title",))
    abstract_col = pick_column(df.columns, ("abstract", "abstract_text"))
    label_col = pick_column(
        df.columns,
        ("label_included", "included", "label", "relevant", "is_relevant"),
    )
    id_col = pick_column(df.columns, ("id", "record_id", "openalex_id", "doi"))
    year_col = pick_column(df.columns, ("publication_year", "year", "published_year"))
    publisher_col = pick_column(df.columns, ("publisher", "venue", "source"))

    if title_col is None or abstract_col is None or label_col is None:
        raise ValueError(
            "Input file must contain title, abstract, and inclusion-label columns. "
            "Expected columns similar to: title, abstract, label_included."
        )

    if topic and topic_col is None:
        raise ValueError(
            "The input file does not contain a topic/dataset column, so --topic cannot be used."
        )

    if topic and topic_col is not None:
        df = df[df[topic_col].astype(str).str.strip().str.lower() == topic.strip().lower()]

    if df.empty:
        raise ValueError("No records matched the requested topic filter.")

    df = df.copy()
    df = df[df[title_col].notna() & df[abstract_col].notna()]
    df = df[df[abstract_col].astype(str).str.strip() != ""]

    records: List[SampleRecord] = []
    for idx, row in df.iterrows():
        record_id = str(row[id_col]).strip() if id_col else str(idx)
        records.append(
            SampleRecord(
                record_id=record_id or str(idx),
                title=str(row[title_col]).strip(),
                abstract=str(row[abstract_col]).strip(),
                label_included=normalize_label(row[label_col]),
                year=str(row[year_col]).strip() if year_col and not pd.isna(row[year_col]) else "",
                publisher=(
                    str(row[publisher_col]).strip()
                    if publisher_col and not pd.isna(row[publisher_col])
                    else ""
                ),
                source_dataset=(
                    str(row[topic_col]).strip()
                    if topic_col and not pd.isna(row[topic_col])
                    else ""
                ),
            )
        )
    return records


def sample_balanced_records(
    records: Sequence[SampleRecord],
    sample_size: int,
    seed: int,
) -> List[SampleRecord]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    included = [record for record in records if record.label_included == 1]
    excluded = [record for record in records if record.label_included == 0]

    target_included = min(len(included), sample_size // 2)
    target_excluded = min(len(excluded), sample_size - target_included)

    if target_included + target_excluded < sample_size:
        remaining = sample_size - (target_included + target_excluded)
        extra_included = max(0, min(len(included) - target_included, remaining))
        target_included += extra_included
        remaining -= extra_included
        extra_excluded = max(0, min(len(excluded) - target_excluded, remaining))
        target_excluded += extra_excluded

    if target_included + target_excluded < sample_size:
        raise ValueError(
            f"Requested {sample_size} records but only {target_included + target_excluded} "
            "usable labeled records are available."
        )

    included_df = pd.DataFrame([asdict(record) for record in included])
    excluded_df = pd.DataFrame([asdict(record) for record in excluded])

    sampled_rows: List[Dict[str, Any]] = []
    if target_included:
        sampled_rows.extend(
            included_df.sample(n=target_included, random_state=seed).to_dict("records")
        )
    if target_excluded:
        sampled_rows.extend(
            excluded_df.sample(n=target_excluded, random_state=seed + 1).to_dict("records")
        )

    sampled_df = pd.DataFrame(sampled_rows).sample(frac=1.0, random_state=seed + 2)
    return [SampleRecord(**row) for row in sampled_df.to_dict("records")]


def strict_format_compliance(raw: str, criteria_count: int) -> int:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != criteria_count:
        return 0
    seen = set()
    valid_answers = {"YES", "NO", "INSUFFICIENT", "INCLUDED", "EXCLUDED"}
    for line in lines:
        if ":" not in line:
            return 0
        left, right = line.split(":", 1)
        left = left.strip().upper()
        right = right.strip().upper()
        if not left.startswith("C"):
            return 0
        try:
            index = int(left[1:])
        except ValueError:
            return 0
        if index < 1 or index > criteria_count or index in seen:
            return 0
        answer = right.split()[0] if right else ""
        if answer not in valid_answers:
            return 0
        seen.add(index)
    return int(len(seen) == criteria_count)


def screen_once(
    record: SampleRecord,
    criteria: List[str],
    temperature: float,
) -> TrialResult:
    paper = {
        "title": record.title,
        "abstract": record.abstract,
        "year": record.year,
        "publisher": record.publisher,
    }
    prompt = build_prompt(paper, criteria, SCREEN_INITIAL_PROMPT)
    raw = call_gpt_chat(
        messages=[
            {"role": "system", "content": "You are an expert SLR screener assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=800,
    )
    parsed = parse_screening_answers(raw, criteria)
    score = relevance_score(parsed)
    counts = count_answers(parsed)
    predicted_label = int(score > 0)

    return TrialResult(
        temperature=temperature,
        repeat_index=0,
        record_id=record.record_id,
        gold_label=record.label_included,
        predicted_label=predicted_label,
        relevance_score=score,
        include_yes=counts["include"]["yes"],
        include_no=counts["include"]["no"],
        include_insufficient=counts["include"]["insufficient"],
        exclude_yes=counts["exclude"]["yes"],
        exclude_no=counts["exclude"]["no"],
        exclude_insufficient=counts["exclude"]["insufficient"],
        format_compliant=strict_format_compliance(raw, len(criteria)),
        raw=raw,
    )


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_run(run_df: pd.DataFrame) -> Dict[str, Any]:
    tp = int(((run_df["gold_label"] == 1) & (run_df["predicted_label"] == 1)).sum())
    tn = int(((run_df["gold_label"] == 0) & (run_df["predicted_label"] == 0)).sum())
    fp = int(((run_df["gold_label"] == 0) & (run_df["predicted_label"] == 1)).sum())
    fn = int(((run_df["gold_label"] == 1) & (run_df["predicted_label"] == 0)).sum())

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, len(run_df))

    return {
        "n_records": int(len(run_df)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_relevance_score": round(float(run_df["relevance_score"].mean()), 4),
        "format_compliance_rate": round(float(run_df["format_compliant"].mean()), 4),
    }


def summarize_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for temperature, temp_df in df.groupby("temperature"):
        repeats = int(temp_df["repeat_index"].nunique())
        if repeats <= 1:
            rows.append(
                {
                    "temperature": temperature,
                    "repeats": repeats,
                    "decision_stability": 1.0,
                    "score_std_mean": 0.0,
                }
            )
            continue

        pivot_pred = temp_df.pivot(
            index="record_id", columns="repeat_index", values="predicted_label"
        )
        pivot_score = temp_df.pivot(
            index="record_id", columns="repeat_index", values="relevance_score"
        )
        stable = pivot_pred.nunique(axis=1).eq(1).mean()
        score_std = pivot_score.std(axis=1, ddof=0).fillna(0.0).mean()
        rows.append(
            {
                "temperature": temperature,
                "repeats": repeats,
                "decision_stability": round(float(stable), 4),
                "score_std_mean": round(float(score_std), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("temperature").reset_index(drop=True)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    criteria_path = Path(args.criteria_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    criteria = load_criteria(criteria_path)
    records = load_synergy_records(input_path, args.topic)
    sample = sample_balanced_records(records, sample_size=args.sample_size, seed=args.seed)

    sampled_df = pd.DataFrame([asdict(record) for record in sample])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_path = out_dir / f"llm_temperature_sample_{timestamp}.csv"
    sampled_df.to_csv(sample_path, index=False, encoding="utf-8")

    all_trials: List[TrialResult] = []
    for temperature in args.temperatures:
        repeats = 1 if temperature == 0.0 else args.nonzero_repeats
        for repeat_index in range(1, repeats + 1):
            print(f"Running temperature={temperature} repeat={repeat_index}/{repeats}")
            for record in sample:
                result = screen_once(record, criteria, temperature=temperature)
                result.repeat_index = repeat_index
                all_trials.append(result)

    trials_df = pd.DataFrame([asdict(item) for item in all_trials])

    summary_rows: List[Dict[str, Any]] = []
    for (temperature, repeat_index), run_df in trials_df.groupby(
        ["temperature", "repeat_index"], sort=True
    ):
        summary = summarize_run(run_df.reset_index(drop=True))
        summary["temperature"] = temperature
        summary["repeat_index"] = repeat_index
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["temperature", "repeat_index"]
    ).reset_index(drop=True)
    stability_df = summarize_stability(trials_df)

    aggregate_df = (
        summary_df.groupby("temperature", as_index=False)
        .agg(
            repeats=("repeat_index", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            format_compliance_mean=("format_compliance_rate", "mean"),
        )
        .round(4)
        .merge(stability_df, on="temperature", how="left")
        .sort_values("temperature")
        .reset_index(drop=True)
    )

    trials_path = out_dir / f"llm_temperature_trials_{timestamp}.csv"
    summary_path = out_dir / f"llm_temperature_summary_{timestamp}.csv"
    aggregate_path = out_dir / f"llm_temperature_aggregate_{timestamp}.csv"
    json_path = out_dir / f"llm_temperature_report_{timestamp}.json"

    trials_df.to_csv(trials_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8")

    report = {
        "config": {
            "input": str(input_path),
            "criteria_file": str(criteria_path),
            "topic": args.topic,
            "temperatures": list(args.temperatures),
            "sample_size": args.sample_size,
            "nonzero_repeats": args.nonzero_repeats,
            "seed": args.seed,
        },
        "sample_path": str(sample_path),
        "trials_path": str(trials_path),
        "summary_path": str(summary_path),
        "aggregate_path": str(aggregate_path),
        "aggregate_results": aggregate_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print("Saved outputs:")
    print(f"- Sample: {sample_path}")
    print(f"- Trials: {trials_path}")
    print(f"- Summary: {summary_path}")
    print(f"- Aggregate: {aggregate_path}")
    print(f"- Report: {json_path}")
    print("")
    print("Aggregate results:")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
