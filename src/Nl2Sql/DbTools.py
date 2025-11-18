import json
from typing import Any, Dict, List, Optional

from langchain.tools import tool

# 如果这个文件和 DatabaseClient 在同一个模块，可以直接 from DbUtils import ...
# 如果不在同一个包里，根据你的目录结构自己调一下导入路径
from .DbUtils import DatabaseClient, SqlValidationError


def get_db_tools(db_client: DatabaseClient):
    """
    基于已有的 DatabaseClient，构造一组 LangChain Tools。
    返回值是一个 List[Tool]，可以直接交给 agent 使用。
    """

    # ============= 1. 列出所有表 =============

    @tool("db_list_tables")
    def db_list_tables(schema_name: Optional[str] = None) -> str:
        """
        列出数据库中所有可用的表。可选 schema。
        返回 JSON:
        {
          "ok": true,
          "schema": "...",
          "tables": ["t1", "t2", ...]
        }
        """
        try:
            tables = db_client.list_tables(schema=schema_name)
            return json.dumps(
                {
                    "ok": True,
                    "schema": schema_name or db_client.default_schema,
                    "tables": tables,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "db_error",
                    "message": f"获取表列表失败: {e}",
                },
                ensure_ascii=False,
            )

    # ============= 2. 获取表结构信息（不含 DDL） =============

    @tool("db_get_table_schema")
    def db_get_table_schema(
        table_name: str,
        schema_name: Optional[str] = None,
    ) -> str:
        """
        获取指定表的结构信息（列名、类型、主键、注释等），不包含建表 DDL。
        这是给模型理解 schema 用的。
        返回 JSON:
        {
          "ok": true,
          "table_name": "...",
          "schema": "...",
          "table_comment": "...",
          "columns": [
            {"name": "...", "type": "...", "nullable": true, "is_primary_key": false, "comment": "..."},
            ...
          ]
        }
        """
        try:
            info = db_client.get_table_schema(table_name, schema=schema_name)
            # 如果没有 columns，视为错误
            if not info.get("columns"):
                return json.dumps(
                    {
                        "ok": False,
                        "error_type": "not_found",
                        "message": f"表 '{table_name}' 不存在或无列定义",
                    },
                    ensure_ascii=False,
                )

            info["ok"] = True
            return json.dumps(info, ensure_ascii=False)

        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "db_error",
                    "message": f"获取表结构失败: {e}",
                },
                ensure_ascii=False,
            )

    # ============= 3. 获取建表语句 + 注释 =============

    @tool("db_get_table_ddl")
    def db_get_table_ddl(
        table_name: str,
        schema_name: Optional[str] = None,
    ) -> str:
        """
        获取指定表的 CREATE TABLE 语句以及表/字段注释。
        返回 JSON:
        {
          "ok": true,
          "create_ddl": "...",
          "table_comment": "...",
          "columns": [...]
        }
        """
        try:
            ddl_info = db_client.get_table_ddl_and_comments(
                table_name,
                schema=schema_name,
            )
            ddl_info["ok"] = True
            return json.dumps(ddl_info, ensure_ascii=False)
        except ValueError as e:
            # 你在 DatabaseClient 中对不存在的表抛的是 ValueError
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "not_found",
                    "message": str(e),
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "db_error",
                    "message": f"获取建表语句失败: {e}",
                },
                ensure_ascii=False,
            )

    # ============= 4. 预览表数据 =============

    @tool("db_preview_table")
    def db_preview_table(
        table_name: str,
        limit: int = 20,
        schema_name: Optional[str] = None,
    ) -> str:
        """
        预览指定表的前 N 行数据，帮助模型理解数据分布。
        返回 JSON:
        {
          "ok": true,
          "table": "...",
          "schema": "...",
          "row_count": 10,
          "rows": [ {...}, {...}, ... ]
        }
        """
        try:
            rows = db_client.fetch_table_preview(
                table_name=table_name,
                limit=limit,
                schema=schema_name,
            )
            return json.dumps(
                {
                    "ok": True,
                    "table": table_name,
                    "schema": schema_name or db_client.default_schema,
                    "row_count": len(rows),
                    "rows": rows,
                },
                ensure_ascii=False,
            )
        except SqlValidationError as e:
            # 理论上这里不会触发，但留作防御
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "validation_error",
                    "message": str(e),
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "db_error",
                    "message": f"获取表数据预览失败: {e}",
                },
                ensure_ascii=False,
            )

    # ============= 5. 执行 SELECT / WITH 查询 =============

    @tool("db_execute_query")
    def db_execute_query(sql: str, max_rows: int = 1000) -> str:
        """
        执行一个只读的 SQL 查询（必须是 SELECT / WITH）。
        DatabaseClient 内部已经做了安全校验：禁止 DDL / DML，只允许 DQL。

        返回 JSON:
        {
          "ok": true,
          "sql": "...",
          "row_count": 123,
          "max_rows": 1000,
          "rows": [ {...}, {...}, ... ]
        }
        """
        try:
            rows = db_client.execute_query(sql, params=None, max_rows=max_rows)
            return json.dumps(
                {
                    "ok": True,
                    "sql": sql,
                    "row_count": len(rows),
                    "max_rows": max_rows,
                    "rows": rows,
                },
                ensure_ascii=False,
            )
        except SqlValidationError as e:
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "validation_error",
                    "message": str(e),
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "db_error",
                    "message": f"执行查询失败: {e}",
                },
                ensure_ascii=False,
            )

    # ============= 6. 使用 EXPLAIN 验证 SQL =============

    @tool("db_explain_query")
    def db_explain_query(sql: str) -> str:
        """
        使用 EXPLAIN 验证 SQL 查询的可执行性和大致执行计划。
        只对 SELECT / WITH 语句生效。

        返回 JSON:
        {
          "ok": true,
          "sql": "...",
          "plan": [ {...}, {...}, ... ]
        }
        """
        try:
            plan = db_client.explain_query(sql, params=None)
            return json.dumps(
                {
                    "ok": True,
                    "sql": sql,
                    "plan": plan,
                },
                ensure_ascii=False,
            )
        except SqlValidationError as e:
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "validation_error",
                    "message": str(e),
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "db_error",
                    "message": f"EXPLAIN 执行失败: {e}",
                },
                ensure_ascii=False,
            )

    # 把所有工具以列表形式返回
    return [
        db_list_tables,
        db_get_table_schema,
        db_get_table_ddl,
        db_preview_table,
        db_execute_query,
        db_explain_query,
    ]
