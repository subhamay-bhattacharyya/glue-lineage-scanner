![Commit Activity](https://img.shields.io/github/commit-activity/t/subhamay-bhattacharyya/glue-lineage-scanner)&nbsp;![Last Commit](https://img.shields.io/github/last-commit/subhamay-bhattacharyya/glue-lineage-scanner)&nbsp;![Release Date](https://img.shields.io/github/release-date/subhamay-bhattacharyya/glue-lineage-scanner)&nbsp;![Repo Size](https://img.shields.io/github/repo-size/subhamay-bhattacharyya/glue-lineage-scanner)&nbsp;![File Count](https://img.shields.io/github/directory-file-count/subhamay-bhattacharyya/glue-lineage-scanner)&nbsp;![Issues](https://img.shields.io/github/issues/subhamay-bhattacharyya/glue-lineage-scanner)&nbsp;![Top Language](https://img.shields.io/github/languages/top/subhamay-bhattacharyya/glue-lineage-scanner)&nbsp;![Custom Endpoint](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/bsubhamay/5caae7e2bae8c5a60658cdbd7155f2ca/raw/glue-lineage-scanner.json?)

# GlueLineageAI

LLM-powered data lineage extraction for AWS Glue PySpark jobs—run locally or in CI via GitHub Actions.

---

## What this does

- Extracts **sources, targets, and transformations** from AWS Glue PySpark scripts
- Combines:
  - **Deterministic static analysis** (Python AST) for reliable anchors (reads/writes/sql)
  - **LLM inference** for higher-level semantic lineage (optional)
- Outputs lineage as:
  - JSON artifacts (machine-readable)
  - Mermaid graphs (human-readable) *(optional)*
  - OpenLineage-compatible events *(optional / roadmap)*

---

## Example output

### Lineage JSON (high level)
```json
{
  "assets": [],
  "operations": [],
  "edges": []
}
```

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Run (single file - using codespace terminal)

```bash
python3 -m lineage_app.cli --input glue_jobs/job1.py --outdir out
```

### Run (directory - using codespace terminal)

```bash
python3 -m lineage_app.cli --input glue_jobs --outdir out
```