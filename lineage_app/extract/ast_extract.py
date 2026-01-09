# lineage_app/extract/ast_extract.py
"""
Starter AST-based extractor for AWS Glue PySpark lineage anchors.

Goal (v0.1.0):
- Deterministically detect common Glue/Spark read/write/SQL/temp-view patterns
- Produce JSON-serializable "events" you can feed into an LLM for graph inference

This module does NOT attempt full dataflow analysis. It focuses on "anchors":
- Reads:
  - glueContext.create_dynamic_frame.from_catalog(...)
  - glueContext.create_dynamic_frame.from_options(...)
  - spark.read.format(...).load(...)  (best-effort)
- Writes:
  - glueContext.write_dynamic_frame.from_catalog(...)
  - glueContext.write_dynamic_frame.from_options(...)
- SQL:
  - spark.sql("...")
  - sqlContext.sql("...")
- Temp views:
  - df.createOrReplaceTempView("view")
  - df.createTempView("view")
- Conversions (best-effort):
  - DynamicFrame.toDF()
  - DynamicFrame.fromDF(...)

You can extend `CALL_MATCHERS` and `_extract_*` methods as needed.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# -----------------------------
# Event models (JSON-friendly)
# -----------------------------

@dataclass
class BaseEvent:
    event_type: str
    lineno: int
    col_offset: int = 0
    end_lineno: Optional[int] = None
    end_col_offset: Optional[int] = None
    target_var: Optional[str] = None  # if assigned: x = <call>

    # Extra context you may want later
    function: Optional[str] = None  # fully qualified call name (best-effort)
    raw: Dict[str, Any] = field(default_factory=dict)  # parsed args/kwargs


@dataclass
class ReadEvent(BaseEvent):
    source_type: str = ""  # from_catalog | from_options | spark_read
    source: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WriteEvent(BaseEvent):
    sink_type: str = ""  # to_catalog | to_options
    sink: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SqlEvent(BaseEvent):
    sql: str = ""


@dataclass
class TempViewEvent(BaseEvent):
    view_name: str = ""
    df_var: Optional[str] = None  # best-effort: name of df var if called on a Name


@dataclass
class ConvertEvent(BaseEvent):
    convert_type: str = ""  # toDF | fromDF
    input_var: Optional[str] = None
    output_var: Optional[str] = None


# -----------------------------
# Utilities
# -----------------------------

def _node_span(node: ast.AST) -> Tuple[int, int, Optional[int], Optional[int]]:
    lineno = getattr(node, "lineno", -1)
    col = getattr(node, "col_offset", 0)
    end_lineno = getattr(node, "end_lineno", None)
    end_col = getattr(node, "end_col_offset", None)
    return lineno, col, end_lineno, end_col


def _is_str_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _get_str_const(node: ast.AST) -> Optional[str]:
    if _is_str_const(node):
        return node.value  # type: ignore[return-value]
    return None


def _literal(node: ast.AST) -> Any:
    """
    Best-effort evaluation of literals.
    Returns a Python value if safe/obvious; otherwise returns a sentinel string.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            out[_literal(k)] = _literal(v)
        return out
    if isinstance(node, ast.List):
        return [_literal(x) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(x) for x in node.elts)
    if isinstance(node, ast.Name):
        return {"$ref": node.id}
    if isinstance(node, ast.Attribute):
        # e.g., args.DB_NAME
        return {"$ref": _full_attr_name(node)}
    if isinstance(node, ast.JoinedStr):
        # f"...{x}..."
        return "<f-string>"
    if isinstance(node, ast.BinOp):
        # string concat or path concat; don't evaluate
        return "<expr>"
    if isinstance(node, ast.Call):
        return "<call>"
    return "<non-literal>"


def _full_attr_name(node: ast.AST) -> str:
    """
    Convert nested ast.Attribute chains to dotted string, best-effort.
    """
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    return ".".join(parts)


def _call_name(node: ast.AST) -> str:
    """
    Return fully qualified call name for a node.func, best-effort.
    Examples:
      glueContext.create_dynamic_frame.from_catalog
      spark.sql
      df.createOrReplaceTempView
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _full_attr_name(node)
    return "<unknown>"


def _extract_kwargs(call: ast.Call) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg:
            out[kw.arg] = _literal(kw.value)
        else:
            # **kwargs
            out["**"] = _literal(kw.value)
    return out


def _extract_args(call: ast.Call) -> List[Any]:
    return [_literal(a) for a in call.args]


# -----------------------------
# Visitor
# -----------------------------

class GlueLineageVisitor(ast.NodeVisitor):
    """
    Collects anchor events.
    """

    # "Matchers" are handled in visit_Call by checking suffix/prefix patterns.
    READ_SUFFIXES = (
        "create_dynamic_frame.from_catalog",
        "create_dynamic_frame.from_options",
    )
    WRITE_SUFFIXES = (
        "write_dynamic_frame.from_catalog",
        "write_dynamic_frame.from_options",
    )
    SQL_CALLS = ("spark.sql", "sqlContext.sql")
    TEMP_VIEW_CALLS = ("createOrReplaceTempView", "createTempView")
    CONVERT_SUFFIXES = ("toDF",)  # DynamicFrame.toDF
    FROM_DF_SUFFIX = "DynamicFrame.fromDF"  # call form: DynamicFrame.fromDF(df, glueContext, "name")

    def __init__(self) -> None:
        self.reads: List[ReadEvent] = []
        self.writes: List[WriteEvent] = []
        self.sqls: List[SqlEvent] = []
        self.temp_views: List[TempViewEvent] = []
        self.conversions: List[ConvertEvent] = []

        # lineno -> assigned var name (best-effort)
        self._assigned_at_line: Dict[int, str] = {}

        # track simple assignments: a = b (Name to Name) (optional)
        self._simple_aliases: Dict[str, str] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        # Track target var for calls at same lineno
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            tgt = node.targets[0].id
            self._assigned_at_line[getattr(node, "lineno", -1)] = tgt

            # Track a = b aliasing (best-effort)
            if isinstance(node.value, ast.Name):
                self._simple_aliases[tgt] = node.value.id

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Handle: x: DataFrame = <call>
        if isinstance(node.target, ast.Name):
            tgt = node.target.id
            self._assigned_at_line[getattr(node, "lineno", -1)] = tgt

            if isinstance(node.value, ast.Name):
                self._simple_aliases[tgt] = node.value.id

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = _call_name(node.func)
        lineno, col, end_lineno, end_col = _node_span(node)
        target_var = self._assigned_at_line.get(lineno)

        # 1) SQL
        if fn in self.SQL_CALLS:
            sql = None
            if node.args:
                sql = _get_str_const(node.args[0])
            if sql:
                ev = SqlEvent(
                    event_type="sql",
                    lineno=lineno,
                    col_offset=col,
                    end_lineno=end_lineno,
                    end_col_offset=end_col,
                    target_var=target_var,
                    function=fn,
                    raw={"args": _extract_args(node), "kwargs": _extract_kwargs(node)},
                    sql=sql,
                )
                self.sqls.append(ev)

        # 2) Glue reads
        if any(fn.endswith(sfx) for sfx in self.READ_SUFFIXES):
            kind = "from_catalog" if fn.endswith("from_catalog") else "from_options"
            details = {"args": _extract_args(node), "kwargs": _extract_kwargs(node)}
            src = self._extract_read_source(fn, node)
            ev = ReadEvent(
                event_type="read",
                lineno=lineno,
                col_offset=col,
                end_lineno=end_lineno,
                end_col_offset=end_col,
                target_var=target_var,
                function=fn,
                raw=details,
                source_type=kind,
                source=src,
            )
            self.reads.append(ev)

        # 3) Glue writes
        if any(fn.endswith(sfx) for sfx in self.WRITE_SUFFIXES):
            kind = "to_catalog" if fn.endswith("from_catalog") else "to_options"
            details = {"args": _extract_args(node), "kwargs": _extract_kwargs(node)}
            sink = self._extract_write_sink(fn, node)
            ev = WriteEvent(
                event_type="write",
                lineno=lineno,
                col_offset=col,
                end_lineno=end_lineno,
                end_col_offset=end_col,
                target_var=target_var,
                function=fn,
                raw=details,
                sink_type=kind,
                sink=sink,
            )
            self.writes.append(ev)

        # 4) Spark reads (best-effort): spark.read.format(...).load(...)
        # This shows up as a call to "load" where func is Attribute(... attr="load")
        if fn.endswith(".load") and "spark.read" in fn:
            src = self._extract_spark_read(node)
            if src:
                ev = ReadEvent(
                    event_type="read",
                    lineno=lineno,
                    col_offset=col,
                    end_lineno=end_lineno,
                    end_col_offset=end_col,
                    target_var=target_var,
                    function=fn,
                    raw={"args": _extract_args(node), "kwargs": _extract_kwargs(node)},
                    source_type="spark_read",
                    source=src,
                )
                self.reads.append(ev)

        # 5) Temp views: df.createOrReplaceTempView("x")
        if any(fn.endswith(f".{tv}") or fn == tv for tv in self.TEMP_VIEW_CALLS):
            view_name = None
            if node.args:
                view_name = _get_str_const(node.args[0])
            if view_name:
                df_var = self._caller_name_if_simple(node.func)
                ev = TempViewEvent(
                    event_type="temp_view",
                    lineno=lineno,
                    col_offset=col,
                    end_lineno=end_lineno,
                    end_col_offset=end_col,
                    target_var=target_var,
                    function=fn,
                    raw={"args": _extract_args(node), "kwargs": _extract_kwargs(node)},
                    view_name=view_name,
                    df_var=df_var,
                )
                self.temp_views.append(ev)

        # 6) Conversions: dyf.toDF()
        if any(fn.endswith(f".{sfx}") for sfx in self.CONVERT_SUFFIXES):
            # Usually: df = dyf.toDF()
            input_var = self._caller_name_if_simple(node.func)
            ev = ConvertEvent(
                event_type="convert",
                lineno=lineno,
                col_offset=col,
                end_lineno=end_lineno,
                end_col_offset=end_col,
                target_var=target_var,
                function=fn,
                raw={"args": _extract_args(node), "kwargs": _extract_kwargs(node)},
                convert_type="toDF",
                input_var=input_var,
                output_var=target_var,
            )
            self.conversions.append(ev)

        # 7) Conversions: DynamicFrame.fromDF(df, glueContext, "name")
        if fn.endswith(self.FROM_DF_SUFFIX):
            input_var = None
            if node.args and isinstance(node.args[0], ast.Name):
                input_var = node.args[0].id
            ev = ConvertEvent(
                event_type="convert",
                lineno=lineno,
                col_offset=col,
                end_lineno=end_lineno,
                end_col_offset=end_col,
                target_var=target_var,
                function=fn,
                raw={"args": _extract_args(node), "kwargs": _extract_kwargs(node)},
                convert_type="fromDF",
                input_var=input_var,
                output_var=target_var,
            )
            self.conversions.append(ev)

        self.generic_visit(node)

    # -----------------------------
    # Extractors
    # -----------------------------

    def _extract_read_source(self, fn: str, call: ast.Call) -> Dict[str, Any]:
        """
        Normalize common read parameters into a consistent dict.
        """
        kw = _extract_kwargs(call)

        if fn.endswith("from_catalog"):
            # from_catalog(database=..., table_name=...)
            db = kw.get("database")
            tbl = kw.get("table_name") or kw.get("name")
            return {"kind": "glue_catalog", "database": db, "table": tbl}

        if fn.endswith("from_options"):
            # from_options(connection_type="s3", connection_options={"paths":[...], ...}, format="json")
            ctype = kw.get("connection_type")
            copt = kw.get("connection_options") or {}
            fmt = kw.get("format")
            paths = None
            if isinstance(copt, dict):
                paths = copt.get("paths") or copt.get("path")
            return {"kind": "options", "connection_type": ctype, "format": fmt, "paths": paths, "options": copt}

        return {"kind": "unknown", "function": fn}

    def _extract_write_sink(self, fn: str, call: ast.Call) -> Dict[str, Any]:
        kw = _extract_kwargs(call)

        if fn.endswith("from_catalog"):
            # write_dynamic_frame.from_catalog(database=..., table_name=...)
            db = kw.get("database")
            tbl = kw.get("table_name") or kw.get("name")
            return {"kind": "glue_catalog", "database": db, "table": tbl}

        if fn.endswith("from_options"):
            # write_dynamic_frame.from_options(connection_type="s3", connection_options={"path":...}, format="parquet")
            ctype = kw.get("connection_type")
            copt = kw.get("connection_options") or {}
            fmt = kw.get("format")
            path = None
            if isinstance(copt, dict):
                path = copt.get("path") or copt.get("paths")
            return {"kind": "options", "connection_type": ctype, "format": fmt, "path": path, "options": copt}

        return {"kind": "unknown", "function": fn}

    def _extract_spark_read(self, call: ast.Call) -> Optional[Dict[str, Any]]:
        """
        Best-effort extraction for spark.read...load() chains.
        We can only reliably capture literal args/kwargs in .load(...).
        """
        # load("s3://...") or load(path="...")
        path = None
        if call.args and _is_str_const(call.args[0]):
            path = call.args[0].value  # type: ignore[assignment]
        if "path" in _extract_kwargs(call) and isinstance(_extract_kwargs(call)["path"], str):
            path = _extract_kwargs(call)["path"]

        if path:
            return {"kind": "spark_read", "path": path}
        return None

    def _caller_name_if_simple(self, func_node: ast.AST) -> Optional[str]:
        """
        If the call is like `df.createOrReplaceTempView(...)` or `dyf.toDF()`,
        return "df" / "dyf" when the receiver is a simple Name.
        """
        if isinstance(func_node, ast.Attribute):
            if isinstance(func_node.value, ast.Name):
                name = func_node.value.id
                # resolve simple alias if present
                return self._simple_aliases.get(name, name)
        return None


# -----------------------------
# Public API
# -----------------------------

def extract_events(code: str) -> Dict[str, Any]:
    """
    Parse code and return JSON-serializable events.
    """
    tree = ast.parse(code)
    visitor = GlueLineageVisitor()
    visitor.visit(tree)

    def dump(events: List[BaseEvent]) -> List[Dict[str, Any]]:
        return [asdict(e) for e in events]

    return {
        "reads": dump(visitor.reads),
        "writes": dump(visitor.writes),
        "sql": dump(visitor.sqls),
        "temp_views": dump(visitor.temp_views),
        "conversions": dump(visitor.conversions),
    }
