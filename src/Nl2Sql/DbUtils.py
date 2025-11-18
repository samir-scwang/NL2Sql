from typing import Any, Dict, List, Optional, Sequence

import re
from contextlib import contextmanager

from sqlalchemy import create_engine, text, inspect, MetaData, Table
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import SQLAlchemyError, NoSuchTableError
from sqlalchemy.schema import CreateTable

try:
    import sqlparse  # 可选，用于更稳健解析 SQL
except ImportError:
    sqlparse = None


class SqlValidationError(Exception):
    """SQL 校验失败（非 SELECT / 存在危险语句等）"""
    pass


class DatabaseClient:
    """
    面向 NL2SQL agent 的数据库工具类。
    - 只允许 SELECT / WITH 查询（DQL），禁止 DDL/DML。
    - 提供：表列表、表内容预览、建表语句+注释、表结构信息、执行查询、EXPLAIN 验证。
    """

    def __init__(
        self,
        connection_url: str,
        echo: bool = False,
        default_schema: Optional[str] = None,
    ) -> None:
        """
        :param connection_url: SQLAlchemy 连接串
          例：
            - sqlite:///example.db
            - mysql+pymysql://user:pwd@host:3306/dbname
            - postgresql+psycopg2://user:pwd@host:5432/dbname
        :param echo: 是否打印 SQL
        :param default_schema: 默认 schema（PostgreSQL 等可用）
        """
        self.engine: Engine = create_engine(connection_url, echo=echo, future=True)
        self.default_schema = default_schema
        self._inspector = inspect(self.engine)

    # ========================= 基础工具 =========================

    @property
    def dialect_name(self) -> str:
        return self.engine.dialect.name  # mysql / postgresql / sqlite / ...

    @contextmanager
    def _connect(self):
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    # ========================= SQL 校验 =========================

    def _validate_select_only(self, sql: str) -> None:
        """
        只允许：
        - 单条语句
        - 以 SELECT 或 WITH 开头的查询
        - 不包含危险关键字（DDL/DML）
        校验不通过则抛出 SqlValidationError
        """
        if not sql or not sql.strip():
            raise SqlValidationError("SQL 为空")

        # 去掉首尾空白和末尾分号
        raw_sql = sql.strip().strip(";")

        # 黑名单关键字（粗暴但有效，多加几层保险）
        forbidden_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
            "TRUNCATE", "CREATE", "RENAME", "MERGE",
            "GRANT", "REVOKE", "REPLACE",
        ]
        upper_sql = raw_sql.upper()

        # 多语句快速拦截（简单处理；更细可以交给 sqlparse）
        if ";" in upper_sql:
            raise SqlValidationError("禁止多条语句，只能执行单条 SELECT/WITH 查询")

        # sqlparse 优先
        if sqlparse is not None:
            statements = [s for s in sqlparse.parse(raw_sql) if str(s).strip()]
            if len(statements) != 1:
                raise SqlValidationError("禁止多条语句，只能执行单条 SELECT/WITH 查询")

            stmt = statements[0]
            stmt_type = stmt.get_type()  # SELECT / INSERT / UNKNOWN ...
            # WITH 语句在 sqlparse 中也会被识别为 SELECT
            if stmt_type != "SELECT":
                raise SqlValidationError(f"只允许 SELECT/WITH 查询，当前语句类型为: {stmt_type}")
        else:
            # 无 sqlparse 时，简单判断开头
            lowered = raw_sql.lower()
            if not (lowered.startswith("select") or lowered.startswith("with ")):
                raise SqlValidationError("只允许以 SELECT 或 WITH 开头的查询")

        # 扫描黑名单关键字（避免子句中带 DDL/DML）
        for kw in forbidden_keywords:
            # 用 \b 保证是完整单词
            if re.search(rf"\b{kw}\b", upper_sql):
                raise SqlValidationError(f"检测到禁止关键字: {kw}，只允许纯查询语句 (DQL)")

    # ========================= 公共基础执行 =========================

    def _safe_execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Result:
        """执行已校验的 SQL，外部不要直接调用"""
        self._validate_select_only(sql)
        with self._connect() as conn:
            try:
                result = conn.execute(text(sql), params or {})
                return result
            except SQLAlchemyError as e:
                # 这里可以按需打日志
                raise e

    # ========================= 对外接口 =========================

    # --- 1. 获取表列表 ---

    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        获取指定 schema 下的所有表名。
        :param schema: 不传则使用 default_schema
        """
        target_schema = schema or self.default_schema
        try:
            return self._inspector.get_table_names(schema=target_schema)
        except SQLAlchemyError as e:
            raise RuntimeError(f"获取表列表失败: {e}") from e

    # --- 2. 获取表数据预览 ---

    def fetch_table_preview(
        self,
        table_name: str,
        limit: int = 20,
        schema: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取表的前 N 行数据，用于 agent 预览表内容。
        """
        target_schema = schema or self.default_schema
        full_name = (
            f'"{target_schema}".\"{table_name}\"'
            if target_schema and self.dialect_name != "mysql"
            else f"{table_name}"
        )

        sql = f"SELECT * FROM {full_name} LIMIT :_limit"
        result = self._safe_execute(sql, {"_limit": limit})

        rows = result.mappings().all()  # 返回 RowMapping
        return [dict(r) for r in rows]

    # --- 3. 获取建表语句 + 注释 ---

    def get_table_ddl_and_comments(
        self,
        table_name: str,
        schema: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        使用 SQLAlchemy 反射生成 CREATE TABLE 语句，
        并通过 inspector 获取表/字段注释。
        返回结构：{
          "create_ddl": str,
          "table_comment": str | None,
          "columns": [
            {
              "name": str,
              "type": str,
              "nullable": bool,
              "default": Any,
              "is_primary_key": bool,
              "comment": str | None,
            },
            ...
          ]
        }
        """
        target_schema = schema or self.default_schema

        metadata = MetaData()
        try:
            table = Table(
                table_name,
                metadata,
                schema=target_schema,
                autoload_with=self.engine,
            )
        except NoSuchTableError as e:
            raise ValueError(f"表不存在: {table_name}") from e
        except SQLAlchemyError as e:
            raise RuntimeError(f"反射表结构失败: {e}") from e

        # 生成 DDL（跨数据库通用）
        try:
            ddl = str(CreateTable(table).compile(self.engine))
        except SQLAlchemyError:
            ddl = f"-- 无法生成标准 DDL，表名: {table_name}"

        # 表注释
        try:
            t_comment_info = self._inspector.get_table_comment(
                table_name, schema=target_schema
            )
            table_comment = t_comment_info.get("text")
        except SQLAlchemyError:
            table_comment = None

        # 列信息（包含注释）
        columns_info = self._inspector.get_columns(table_name, schema=target_schema)
        pk_info = self._inspector.get_pk_constraint(table_name, schema=target_schema)
        pk_cols = set(pk_info.get("constrained_columns") or [])

        columns = []
        for col in columns_info:
            columns.append(
                {
                    "name": col.get("name"),
                    "type": str(col.get("type")),
                    "nullable": bool(col.get("nullable", True)),
                    "default": col.get("default"),
                    "is_primary_key": col.get("name") in pk_cols,
                    "comment": col.get("comment"),  # 部分方言支持
                }
            )

        return {
            "create_ddl": ddl,
            "table_comment": table_comment,
            "columns": columns,
        }

    # --- 4. 获取表结构信息（更轻量，不要 DDL 时用这个） ---

    def get_table_schema(
        self,
        table_name: str,
        schema: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        获取表结构信息（不含 DDL），主要给 NL2SQL 模型做 schema 提示用。
        返回结构：
        {
          "table_name": str,
          "schema": str | None,
          "columns": [
            {
              "name": str,
              "type": str,
              "nullable": bool,
              "default": Any,
              "is_primary_key": bool,
              "comment": str | None,
            },
            ...
          ],
          "table_comment": str | None
        }
        """
        target_schema = schema or self.default_schema

        try:
            columns_info = self._inspector.get_columns(table_name, schema=target_schema)
            pk_info = self._inspector.get_pk_constraint(table_name, schema=target_schema)
            pk_cols = set(pk_info.get("constrained_columns") or [])
            t_comment_info = self._inspector.get_table_comment(
                table_name, schema=target_schema
            )
        except SQLAlchemyError as e:
            raise RuntimeError(f"获取表结构失败: {e}") from e

        columns = []
        for col in columns_info:
            columns.append(
                {
                    "name": col.get("name"),
                    "type": str(col.get("type")),
                    "nullable": bool(col.get("nullable", True)),
                    "default": col.get("default"),
                    "is_primary_key": col.get("name") in pk_cols,
                    "comment": col.get("comment"),
                }
            )

        return {
            "table_name": table_name,
            "schema": target_schema,
            "columns": columns,
            "table_comment": t_comment_info.get("text"),
        }

    # --- 5. 执行 SELECT 查询 ---

    def execute_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        max_rows: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        执行 SELECT/WITH 查询。
        - 自动校验：只能 DQL
        - 自动限制最大返回行数
        """
        result = self._safe_execute(sql, params)
        rows = result.mappings().fetchmany(max_rows)
        return [dict(r) for r in rows]

    # --- 6. 使用 EXPLAIN 验证 SQL ---

    def explain_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        使用 EXPLAIN 验证 SQL 的“争取性”和可执行性。
        - 同样会做只允许 DQL 的校验
        - 不实际返回业务数据，只返回执行计划
        """
        # 先验证 SQL 是否安全
        self._validate_select_only(sql)

        dialect = self.dialect_name
        explain_sql: str

        if dialect == "sqlite":
            explain_sql = f"EXPLAIN QUERY PLAN {sql}"
        else:
            # MySQL / PostgreSQL / 其他方言
            explain_sql = f"EXPLAIN {sql}"

        with self._connect() as conn:
            try:
                result = conn.execute(text(explain_sql), params or {})
                rows = result.mappings().all()
                return [dict(r) for r in rows]
            except SQLAlchemyError as e:
                # 如果 EXPLAIN 都报错，可以视为 SQL 不可执行
                raise RuntimeError(f"EXPLAIN 执行失败，SQL 可能有问题: {e}") from e
if __name__ == "__main__":
    # 例：连接本地 MySQL
    db = DatabaseClient(
        "mysql+pymysql://root:sql2008@127.0.0.1:3306/mysales_v2",
        echo=False,
        default_schema=None,  # MySQL 可以不填
    )

    # 1. 列出所有表
    print(db.list_tables())

    # 2. 看某个表前 10 行
    preview = db.fetch_table_preview("orders", limit=10)
    print(preview)

    # 3. 获取建表语句 + 注释
    ddl_info = db.get_table_ddl_and_comments("orders")
    print(ddl_info["create_ddl"])
    print(ddl_info["table_comment"])
    print(ddl_info["columns"])

    # 4. 获取表结构（给 NL2SQL 模型用）
    schema_info = db.get_table_schema("orders")
    print(schema_info)

    # 5. 执行一个 SELECT 查询
    rows = db.execute_query("SELECT *FROM orders", {})
    print(rows)

    # 6. 用 EXPLAIN 验证 SQL
    plan = db.explain_query("SELECT * FROM orders WHERE orderid = 1")
    print(plan)
