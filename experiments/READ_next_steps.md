# Temperature Experiment: Next Steps

This guide explains exactly what to do next to run the quick SYNERGY-based temperature experiment in `experiments/llm_temperature.py`.

## 1. Get the SYNERGY dataset file

Download the SYNERGY dataset from:

- https://github.com/asreview/synergy-dataset

What you need:

- A local data file in `CSV`, `JSON`, or `JSONL` format
- The file should contain columns similar to:
  - `title`
  - `abstract`
  - `label_included` or `included` or `label`
  - optionally a review/topic column such as `dataset`, `review`, or `topic`

Recommended local path:

```text
experiments/data/synergy.csv
```

If the repository provides several files, use either:

- one combined export containing multiple review topics, or
- one single-topic file

## 2. Create the criteria file

Create this file:

```text
experiments/data/criteria.txt
```

Put one criterion per line. Keep the criteria fixed for the whole experiment.

Example:

```text
INCLUDE: Does the study evaluate an intervention, method, or system relevant to the review question?
INCLUDE: Does the title or abstract describe the target population or domain of interest?
INCLUDE: Does the study report empirical results, experiments, or evaluation findings?
EXCLUDE: Is the study clearly unrelated to the review question?
EXCLUDE: Is the paper a protocol, editorial, abstract-only item, or other non-primary study?
EXCLUDE: Does the title or abstract provide insufficient evidence of relevance to the review question?
```

Important:

- Use criteria that match the specific SYNERGY review topic you selected.
- Do not regenerate or edit the criteria between temperatures.

## 3. Inspect the dataset columns

Run this to see the available columns:

```powershell
python -c "import pandas as pd; df=pd.read_csv(r'experiments\data\synergy.csv'); print(df.columns.tolist()); print(df.head())"
```

Check that the file includes:

- a title column
- an abstract column
- an inclusion label column

If your file is JSON instead of CSV, adjust the path in the command and use the file format you downloaded.

## 4. Find the topic name

If the dataset mixes multiple review topics, inspect the topic column so you can choose one review.

Example:

```powershell
python -c "import pandas as pd; df=pd.read_csv(r'experiments\data\synergy.csv'); print(df['dataset'].dropna().unique()[:50])"
```

If your topic column is not named `dataset`, replace it with the actual column name from Step 3.

Pick one topic and keep it fixed for the experiment.

## 5. Run the experiment

Run this from the repo root:

```powershell
python experiments\llm_temperature.py `
  --input experiments\data\synergy.csv `
  --criteria-file experiments\data\criteria.txt `
  --topic Hall_2012 `
  --sample-size 20 `
  --temperatures 0.0 0.2 0.5 1.0 `
  --nonzero-repeats 3
```

What this does:

- samples the same `20` records once
- tries to balance included and excluded records
- runs `1` pass at `temperature=0.0`
- runs `3` passes for each nonzero temperature
- saves trial-level and summary outputs

## 6. Check the output files

After the run, look in:

```text
experiments/results/
```

You should get:

- `llm_temperature_sample_...csv`
- `llm_temperature_trials_...csv`
- `llm_temperature_summary_...csv`
- `llm_temperature_aggregate_...csv`
- `llm_temperature_report_...json`

## 7. What to report in the paper

Use the aggregate file and report:

- accuracy
- precision
- recall
- F1
- format compliance rate
- decision stability across repeated runs

Recommended wording:

```text
We conducted a pilot temperature-ablation study on a fixed 20-record sample from a public SYNERGY review dataset. The same records and screening criteria were used across all conditions. Temperature 0.0 was run once, while nonzero temperatures were run three times to assess variability. We compared screening accuracy, F1, output-format compliance, and run-to-run stability.
```

## 8. If the script fails

Common causes:

- the dataset file does not contain the expected column names
- the topic name does not exactly match the dataset value
- the criteria file is empty
- API credentials are not set in `.env`

Check:

- `GPT_ENDPOINT`
- `GPT_DEPLOYMENT`
- `GPT_KEY`

## 9. Recommended minimal workflow

1. Download SYNERGY into `experiments/data/`
2. Create `criteria.txt`
3. Inspect columns
4. Inspect topic names
5. Run the experiment
6. Use the aggregate CSV for your paper table
