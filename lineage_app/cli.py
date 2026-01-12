# lineage_app/cli.py
import argparse
import json
from pathlib import Path

from lineage_app.extract.ast_extract import extract_events  # your function

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="File or directory to scan")
    p.add_argument("--outdir", default="out", help="Output directory")
    args = p.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = []
    if in_path.is_file():
        files = [in_path]
    else:
        files = sorted(in_path.rglob("*.py"))

    results = []
    for f in files:
        code = f.read_text(encoding="utf-8")
        events = extract_events(code)
        results.append({"file": str(f), "events": events})

    (outdir / "lineage-events.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {outdir/'lineage-events.json'}")

if __name__ == "__main__":
    main()
