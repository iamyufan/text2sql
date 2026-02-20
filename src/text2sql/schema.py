"""Schema serialization: Spider tables.json -> CREATE TABLE string."""

# Spider column_types -> SQLite type
SPIDER_TYPE_TO_SQL: dict[str, str] = {
    "text": "TEXT",
    "number": "INT",
    "time": "TEXT",
    "boolean": "TEXT",
    "others": "TEXT",
}


def _sql_type(spider_type: str) -> str:
    return SPIDER_TYPE_TO_SQL.get(spider_type, "TEXT")


def serialize_schema(schema: dict) -> str:
    """Turn one Spider schema (one db from tables.json) into CREATE TABLE statements.

    Uses table_names_original, column_names_original, column_types, primary_keys,
    foreign_keys. Output is semicolon-separated CREATE TABLE ... ; statements.
    """
    table_names_original = schema["table_names_original"]
    # list of [table_idx, col_name]
    column_names_original = schema["column_names_original"]
    column_types = schema["column_types"]
    primary_keys = set(schema.get("primary_keys", []))
    foreign_keys = schema.get("foreign_keys", [])  # list of [cid, ref_cid]

    # column index -> (table_idx, column_name) for non-(−1, "*")
    col_index_to_table_col: list[tuple[int, str] | None] = []
    for _, (tid, cname) in enumerate(column_names_original):
        if tid == -1 and cname == "*":
            col_index_to_table_col.append(None)
        else:
            col_index_to_table_col.append((tid, cname))

    # Build (table_idx, column_index) for each table's columns (skip index 0 = *).
    table_columns: dict[int, list[int]] = {}  # table_idx -> list of column indices
    for cid in range(1, len(column_names_original)):
        entry = col_index_to_table_col[cid]
        if entry is None:
            continue
        tid, _ = entry
        table_columns.setdefault(tid, []).append(cid)

    # FK: (cid, ref_cid) -> we need (table_name, col_name) and (ref_table_name, ref_col_name)
    def col_index_to_names(cid: int) -> tuple[str, str] | None:
        if cid <= 0 or cid >= len(col_index_to_table_col):
            return None
        entry = col_index_to_table_col[cid]
        if entry is None:
            return None
        tid, cname = entry
        tname = table_names_original[tid]
        return (tname, cname)

    statements = []
    for table_idx, table_name in enumerate(table_names_original):
        col_indices = table_columns.get(table_idx, [])
        parts = []
        for cid in col_indices:
            _, col_name = col_index_to_table_col[cid]
            sql_t = _sql_type(column_types[cid])
            pk_suffix = " PRIMARY KEY" if cid in primary_keys else ""
            parts.append(f"{col_name} {sql_t}{pk_suffix}")

        # Add FOREIGN KEY constraints for columns in this table (child side)
        for cid, ref_cid in foreign_keys:
            if cid not in col_indices:
                continue
            ref_names = col_index_to_names(ref_cid)
            from_names = col_index_to_names(cid)
            if ref_names is None or from_names is None:
                continue
            ref_table, ref_col = ref_names
            from_table, from_col = from_names
            if from_table != table_name:
                continue
            parts.append(f"FOREIGN KEY ({from_col}) REFERENCES {ref_table}({ref_col})")

        if not parts:
            continue
        stmt = f"CREATE TABLE {table_name} (" + ", ".join(parts) + ");"
        statements.append(stmt)

    return " ".join(statements)


def get_schema_for_db(db_id: str, tables_list: list[dict]) -> dict | None:
    """Return the schema dict for the given db_id from the tables.json list."""
    for t in tables_list:
        if t.get("db_id") == db_id:
            return t
    return None


def serialize_schema_for_db(db_id: str, tables_list: list[dict]) -> str:
    """Return the CREATE TABLE string for the given db_id. Raises if not found."""
    schema = get_schema_for_db(db_id, tables_list)
    if schema is None:
        raise KeyError(f"db_id not found in tables: {db_id}")
    return serialize_schema(schema)
