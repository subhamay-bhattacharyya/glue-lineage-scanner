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
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
    sink_type: str = ""  # to_catalog | to_options | spark_write
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


def _full_attr_name(node: ast.AST) -> str:
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    return ".".join(parts)


def _literal(node: ast.AST) -> Any:
    """
    Best-effort evaluation of literals.
    Returns a Python value if safe/obvious; otherwise returns a sentinel string
    or a reference dict like {"$ref": "..."}.
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
        # e.g., glueContext.spark_session
        return {"$ref": _full_attr_name(node)}

    if isinstance(node, ast.Subscript):
        # Example: args["SRC_CUSTOMERS"]
        base = _literal(node.value)

        # Python 3.9+: slice is node.slice
        key = None
        if isinstance(node.slice, ast.Constant):
            key = node.slice.value
        else:
            key = _literal(node.slice)

        if isinstance(base, dict) and "$ref" in base:
            base_str = base["$ref"]
        else:
            base_str = str(base)

        return {"$ref": f"{base_str}[{key!r}]"}

    if isinstance(node, ast.JoinedStr):
        # f"...{x}..."
        return "<f-string>"

    if isinstance(node, ast.BinOp):
        # string concat or path concat; don't evaluate
        return "<expr>"

    if isinstance(node, ast.Call):
        return "<call>"

    return "<non-literal>"


def _call_name(node: ast.AST) -> str:
    """
    Return best-effort dotted call name.
    Supports chained calls where the receiver is itself a Call, e.g.
      spark.read.format(...).load(...)
    """
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        base = node.value
        # If base is a Call (method chaining), recurse into base.func
        if isinstance(base, ast.Call):
            return f"{_call_name(base.func)}.{node.attr}"
        # Normal attribute chain (x.y.z)
        if isinstance(base, (ast.Name, ast.Attribute)):
            return f"{_call_name(base)}.{node.attr}"
        return node.attr

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
    Collects anchor events + variable assignments.
    """

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
    CONVERT_SUFFIXES = ("toDF",)
    FROM_DF_SUFFIX = "DynamicFrame.fromDF"

    def __init__(self) -> None:
        self.reads: List[ReadEvent] = []
        self.writes: List[WriteEvent] = []
        self.sqls: List[SqlEvent] = []
        self.temp_views: List[TempViewEvent] = []
        self.conversions: List[ConvertEvent] = []

        # capture variable values for resolving $ref later
        self.variables: Dict[str, Any] = {}

        # lineno -> assigned var name (best-effort)
        self._assigned_at_line: Dict[int, str] = {}

        # track simple assignments: a = b (Name to Name)
        self._simple_aliases: Dict[str, str] = {}

    def _index_calls_for_assignment(self, expr: ast.AST, target: str) -> None:
        """
        For multi-line chained calls, the Call nodes may be on different line numbers
        than the Assign node. Map all nested Call.lineno -> assigned target var.
        """
        for n in ast.walk(expr):
            if isinstance(n, ast.Call):
                ln = getattr(n, "lineno", None)
                if ln is not None:
                    self._assigned_at_line.setdefault(ln, target)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            tgt = node.targets[0].id

            # map assign lineno
            self._assigned_at_line[getattr(node, "lineno", -1)] = tgt

            # NEW: map nested call line numbers to the same target (chained calls)
            self._index_calls_for_assignment(node.value, tgt)

            # capture variable value (best-effort)
            self.variables[tgt] = _literal(node.value)

            # Track a = b aliasing (best-effort)
            if isinstance(node.value, ast.Name):
                self._simple_aliases[tgt] = node.value.id

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Handle: x: DataFrame = <call> or x: str = "..."
        if isinstance(node.target, ast.Name):
            tgt = node.target.id
            self._assigned_at_line[getattr(node, "lineno", -1)] = tgt

            if node.value is not None:
                # NEW: map nested call line numbers too
                self._index_calls_for_assignment(node.value, tgt)
                self.variables[tgt] = _literal(node.value)

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
                self.sqls.append(
                    SqlEvent(
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
                )

        # 2) Glue reads
        if any(fn.endswith(sfx) for sfx in self.READ_SUFFIXES):
            kind = "from_catalog" if fn.endswith("from_catalog") else "from_options"
            src = self._extract_read_source(fn, node)
            self.reads.append(
                ReadEvent(
                    event_type="read",
                    lineno=lineno,
                    col_offset=col,
                    end_lineno=end_lineno,
                    end_col_offset=end_col,
                    target_var=target_var,
                    function=fn,
                    raw={"args": _extract_args(node), "kwargs": _extract_kwargs(node)},
                    source_type=kind,
                    source=src,
                )
            )

        # 3) Glue writes
        if any(fn.endswith(sfx) for sfx in self.WRITE_SUFFIXES):
            kind = "to_catalog" if fn.endswith("from_catalog") else "to_options"
            sink = self._extract_write_sink(fn, node)
            self.writes.append(
                WriteEvent(
                    event_type="write",
                    lineno=lineno,
                    col_offset=col,
                    end_lineno=end_lineno,
                    end_col_offset=end_col,
                    target_var=target_var,
                    function=fn,
                    raw={"args": _extract_args(node), "kwargs": _extract_kwargs(node)},
                    sink_type=kind,
                    sink=sink,
                )
            )

        # 4) Temp views
        if any(fn.endswith(f".{tv}") or fn == tv for tv in self.TEMP_VIEW_CALLS):
            view_name = None
            if node.args:
                view_name = _get_str_const(node.args[0])
            if view_name:
                df_var = self._caller_name_if_simple(node.func)
                self.temp_views.append(
                    TempViewEvent(
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
                )

        # 5) Conversions: dyf.toDF()
        if any(fn.endswith(f".{sfx}") for sfx in self.CONVERT_SUFFIXES):
            input_var = self._caller_name_if_simple(node.func)
            self.conversions.append(
                ConvertEvent(
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
            )

        # 6) Conversions: DynamicFrame.fromDF(df, glueContext, "name")
        if fn.endswith(self.FROM_DF_SUFFIX):
            input_var = None
            if node.args and isinstance(node.args[0], ast.Name):
                input_var = node.args[0].id
            self.conversions.append(
                ConvertEvent(
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
            )

        # 7) Spark reads: ...load(...)
        if fn.endswith(".load"):
            src = self._extract_spark_load(call=node)
            if src:
                self.reads.append(
                    ReadEvent(
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
                )

        # 8) Spark writes: ...save(...) / ...saveAsTable(...)
        if fn.endswith(".save") or fn.endswith(".saveAsTable"):
            sink = self._extract_spark_save(call=node, fn=fn)
            if sink:
                self.writes.append(
                    WriteEvent(
                        event_type="write",
                        lineno=lineno,
                        col_offset=col,
                        end_lineno=end_lineno,
                        end_col_offset=end_col,
                        target_var=target_var,
                        function=fn,
                        raw={"args": _extract_args(node), "kwargs": _extract_kwargs(node)},
                        sink_type="spark_write",
                        sink=sink,
                    )
                )

        self.generic_visit(node)

    # -----------------------------
    # Extractors
    # -----------------------------

    def _extract_read_source(self, fn: str, call: ast.Call) -> Dict[str, Any]:
        kw = _extract_kwargs(call)

        if fn.endswith("from_catalog"):
            db = kw.get("database")
            tbl = kw.get("table_name") or kw.get("name")
            return {"kind": "glue_catalog", "database": db, "table": tbl}

        if fn.endswith("from_options"):
            ctype = kw.get("connection_type")
            copt = kw.get("connection_options") or {}
            fmt = kw.get("format")
            paths = None
            if isinstance(copt, dict):
                paths = copt.get("paths") or copt.get("path")
            return {
                "kind": "options",
                "connection_type": ctype,
                "format": fmt,
                "paths": paths,
                "options": copt,
            }

        return {"kind": "unknown", "function": fn}

    def _extract_write_sink(self, fn: str, call: ast.Call) -> Dict[str, Any]:
        kw = _extract_kwargs(call)

        if fn.endswith("from_catalog"):
            db = kw.get("database")
            tbl = kw.get("table_name") or kw.get("name")
            return {"kind": "glue_catalog", "database": db, "table": tbl}

        if fn.endswith("from_options"):
            ctype = kw.get("connection_type")
            copt = kw.get("connection_options") or {}
            fmt = kw.get("format")
            path = None
            if isinstance(copt, dict):
                path = copt.get("path") or copt.get("paths")
            return {
                "kind": "options",
                "connection_type": ctype,
                "format": fmt,
                "path": path,
                "options": copt,
            }

        return {"kind": "unknown", "function": fn}

    def _caller_name_if_simple(self, func_node: ast.AST) -> Optional[str]:
        if isinstance(func_node, ast.Attribute):
            if isinstance(func_node.value, ast.Name):
                name = func_node.value.id
                return self._simple_aliases.get(name, name)
        return None

    def _extract_spark_load(self, call: ast.Call) -> Optional[Dict[str, Any]]:
        """
        Best-effort extraction for Spark load(path).
        Captures path as literal or {"$ref": var}.
        """
        path = None
        if call.args:
            path = _literal(call.args[0])
        else:
            kw = _extract_kwargs(call)
            if "path" in kw:
                path = kw["path"]

        if path is None:
            return None

        return {"kind": "spark_load", "path": path}

    def _extract_spark_save(self, call: ast.Call, fn: str) -> Optional[Dict[str, Any]]:
        """
        Best-effort extraction for Spark save/saveAsTable.
        Captures target as literal or {"$ref": var}.
        """
        target = None
        if call.args:
            target = _literal(call.args[0])
        else:
            kw = _extract_kwargs(call)
            target = kw.get("path") or kw.get("tableName")

        if target is None:
            return None

        kind = "saveAsTable" if fn.endswith(".saveAsTable") else "save"
        return {"kind": kind, "target": target}


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
        "variables": visitor.variables,
        "reads": dump(visitor.reads),
        "writes": dump(visitor.writes),
        "sql": dump(visitor.sqls),
        "temp_views": dump(visitor.temp_views),
        "conversions": dump(visitor.conversions),
    }
