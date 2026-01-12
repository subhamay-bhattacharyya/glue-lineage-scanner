# lineage_app/build/build_lineage.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------

def _is_ref(x: Any) -> bool:
    return isinstance(x, dict) and "$ref" in x and isinstance(x["$ref"], str)


def _display_name(x: Any) -> str:
    if _is_ref(x):
        return x["$ref"]
    return str(x)


def _resolve_ref(value: Any, variables: Dict[str, Any]) -> Any:
    """
    Resolve {"$ref": "name"} using variables map.
    Keeps nested refs like args['X'] intact.
    """
    if _is_ref(value):
        return variables.get(value["$ref"], value)
    return value


# -----------------------------
# Lineage builder
# -----------------------------

def build_lineage_for_file(events: Dict[str, Any], file_path: str) -> Dict[str, Any]:
    variables: Dict[str, Any] = events.get("variables", {})

    assets: List[Dict[str, Any]] = []
    operations: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    asset_index: Dict[Tuple[str, str], str] = {}
    asset_counter = 1
    op_counter = 1

    def ensure_asset(asset_type: str, name: Any) -> str:
        nonlocal asset_counter
        display = _display_name(name)
        key = (asset_type, display)
        if key in asset_index:
            return asset_index[key]
        aid = f"a{asset_counter}"
        asset_counter += 1
        assets.append({"id": aid, "type": asset_type, "name": display})
        asset_index[key] = aid
        return aid

    def new_op(op_type: str, tool: str, meta: Optional[Dict[str, Any]] = None) -> str:
        nonlocal op_counter
        oid = f"op{op_counter}"
        op_counter += 1
        op = {"id": oid, "type": op_type, "tool": tool}
        if meta:
            op.update(meta)
        operations.append(op)
        return oid

    # v0.1.0: single transform op per file
    transform_op = new_op(
        op_type="transform",
        tool="pyspark-job",
        meta={"name": Path(file_path).name},
    )

    # Reads: dataset -> transform
    for r in events.get("reads", []):
        path = _resolve_ref(r.get("source", {}).get("path"), variables)
        src_asset = ensure_asset("dataset", path)
        edges.append(
            {
                "from": src_asset,
                "to": transform_op,
                "meta": {
                    "lineno": r.get("lineno"),
                    "df": r.get("target_var"),
                },
            }
        )

    # Writes: transform -> dataset
    for w in events.get("writes", []):
        target = _resolve_ref(w.get("sink", {}).get("target"), variables)
        tgt_asset = ensure_asset("dataset", target)
        edges.append(
            {
                "from": transform_op,
                "to": tgt_asset,
                "meta": {
                    "lineno": w.get("lineno"),
                    "df": "df_out",
                },
            }
        )

    return {"assets": assets, "operations": operations, "edges": edges}


# -----------------------------
# Mermaid rendering
# -----------------------------

def lineage_to_mermaid(lineage: Dict[str, Any]) -> str:
    """
    Convert lineage graph to Mermaid flowchart (GitHub-safe).
    """
    assets = {a["id"]: a for a in lineage["assets"]}
    ops = {o["id"]: o for o in lineage["operations"]}

    lines: List[str] = []
    lines.append("flowchart LR")

    # Assets
    for a in assets.values():
        label = f"{a['type']}<br/>{a['name']}"
        lines.append(f'  {a["id"]}["{label}"]')

    # Operations
    for o in ops.values():
        label = f"{o['type']}<br/>{o.get("name", o.get("tool", ""))}"
        lines.append(f'  {o["id"]}(["{label}"])')

    # Edges
    for e in lineage["edges"]:
        src = e["from"]
        tgt = e["to"]
        meta = e.get("meta", {})
        edge_label = ""
        if meta.get("df"):
            edge_label = f'|{meta["df"]}|'
        lines.append(f"  {src} -->{edge_label} {tgt}")

    return "\n".join(lines)



# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=True, help="Input lineage-events.json")
    p.add_argument("--outfile", required=True, help="Output lineage graph JSON")
    p.add_argument(
        "--mermaid",
        required=False,
        help="Optional Mermaid markdown output (GitHub-friendly). Example: out/lineage.mmd",
    )
    args = p.parse_args()

    docs = json.loads(Path(args.infile).read_text(encoding="utf-8"))

    out: List[Dict[str, Any]] = []

    # Build lineage per file
    for item in docs:
        file_path = item["file"]
        events = item["events"]
        lineage = build_lineage_for_file(events, file_path)
        out.append({"file": file_path, "lineage": lineage})

    # Write JSON output
    Path(args.outfile).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.outfile}")

    # Write Mermaid output (GitHub-friendly)
    if args.mermaid:
        mermaid_lines: List[str] = []
        mermaid_lines.append("```mermaid")
        for item in out:
            file_path = item["file"]
            lineage = item["lineage"]

            # Keep comment, but avoid blank line afterwards (some renderers are picky)
            mermaid_lines.append(f"%% {file_path}")
            mermaid_lines.append(lineage_to_mermaid(lineage))
        mermaid_lines.append("```")

        Path(args.mermaid).write_text("\n".join(mermaid_lines) + "\n", encoding="utf-8")
        print(f"Wrote {args.mermaid}")


if __name__ == "__main__":
    main()
