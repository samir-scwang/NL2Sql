from typing import Optional
import json
import sys
from pathlib import Path

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool, ToolRuntime  # ✅ 引入 ToolRuntime

# ========== 兼容包导入与脚本执行 ==========

if __package__:
    # 作为包导入时的写法，如: from Nl2Sql import agent
    from . import myllm
    from .DbUtils import DatabaseClient
    from .DbTools import get_db_tools
else:
    # 直接 python src/Nl2Sql/agent.py 执行时的兜底写法
    _MODULE_DIR = Path(__file__).resolve().parent
    _SRC_DIR = _MODULE_DIR.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

    from Nl2Sql import myllm  # type: ignore
    from Nl2Sql.DbUtils import DatabaseClient  # type: ignore
    from Nl2Sql.DbTools import get_db_tools  # type: ignore


deepseek = myllm.deepseek


# ================== 1. 自定义 Agent 状态 ==================

class CustomState(AgentState):
    """扩展 AgentState，可以额外保存 user_name 等信息。"""
    user_name: str = ""


# ================== 2. 初始化 DatabaseClient 和 DbTools ==================

db_client = DatabaseClient(
    connection_url="mysql+pymysql://root:sql2008@127.0.0.1:3306/mysales_v2",
    echo=False,
    default_schema=None,
)

# get_db_tools 应该返回一组 LangChain Tool 对象（内部已绑定 db_client）
db_tools = get_db_tools(db_client)
_db_tool_map = {t.name: t for t in db_tools}


def _invoke_db_tool(tool_name: str, **kwargs) -> dict:
    """
    调用 get_db_tools 返回的 LangChain Tool 对象的底层函数，
    并把 JSON 字符串转成 dict。
    """
    tool_obj = _db_tool_map[tool_name]
    result = tool_obj.func(**kwargs)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"ok": False, "error_type": "json_parse_error", "message": result}
    return result


# ================== 3. 基于 DbTools 封装 LangChain Tools（带 ToolRuntime.StreamWriter） ==================

@tool("db_list_tables")
def db_list_tables(
    schema_name: Optional[str] = None,
    runtime: ToolRuntime = None,  # ✅ 类型是 ToolRuntime，默认值 None
) -> dict:
    """
    列出数据库中所有可用的表。
    使用场景：在生成 SQL 前，先了解有哪些表可以用。
    """
    writer = getattr(runtime, "stream_writer", None) if runtime is not None else None
    if writer:
        writer(f"[db_list_tables] 开始执行，schema={schema_name!r}")

    result = _invoke_db_tool("db_list_tables", schema_name=schema_name)

    if writer:
        writer("[db_list_tables] 执行完成")

    return result


@tool("db_get_table_schema")
def db_get_table_schema(
    table_name: str,
    schema_name: Optional[str] = None,
    runtime: ToolRuntime = None,
) -> dict:
    """
    获取指定表的结构信息（列名、类型、主键、注释等），不包含建表 DDL。
    使用场景：在生成 SQL 前，让模型了解某个表的字段。
    """
    writer = getattr(runtime, "stream_writer", None) if runtime is not None else None
    if writer:
        writer(
            f"[db_get_table_schema] 开始执行，table={table_name!r}, schema={schema_name!r}"
        )

    result = _invoke_db_tool(
        "db_get_table_schema",
        table_name=table_name,
        schema_name=schema_name,
    )

    if writer:
        writer("[db_get_table_schema] 执行完成")

    return result


@tool("db_get_table_ddl")
def db_get_table_ddl(
    table_name: str,
    schema_name: Optional[str] = None,
    runtime: ToolRuntime = None,
) -> dict:
    """
    获取指定表的 CREATE TABLE 语句以及表/字段注释。
    使用场景：需要更详细的建表信息时使用。
    """
    writer = getattr(runtime, "stream_writer", None) if runtime is not None else None
    if writer:
        writer(
            f"[db_get_table_ddl] 开始执行，table={table_name!r}, schema={schema_name!r}"
        )

    result = _invoke_db_tool(
        "db_get_table_ddl",
        table_name=table_name,
        schema_name=schema_name,
    )

    if writer:
        writer("[db_get_table_ddl] 执行完成")

    return result


@tool("db_preview_table")
def db_preview_table(
    table_name: str,
    limit: int = 20,
    schema_name: Optional[str] = None,
    runtime: ToolRuntime = None,
) -> dict:
    """
    预览指定表的前 N 行数据，帮助模型理解数据分布。
    """
    writer = getattr(runtime, "stream_writer", None) if runtime is not None else None
    if writer:
        writer(
            f"[db_preview_table] 开始执行，table={table_name!r}, "
            f"schema={schema_name!r}, limit={limit}"
        )

    result = _invoke_db_tool(
        "db_preview_table",
        table_name=table_name,
        limit=limit,
        schema_name=schema_name,
    )

    if writer:
        writer("[db_preview_table] 执行完成")

    return result


@tool("db_explain_query")
def db_explain_query(
    sql: str,
    runtime: ToolRuntime = None,
) -> dict:
    """
    使用 EXPLAIN 验证 SQL 查询的可执行性和执行计划。
    使用场景：模型生成 SQL 后，先检查是否合理、是否能执行。
    """
    writer = getattr(runtime, "stream_writer", None) if runtime is not None else None
    if writer:
        writer(f"[db_explain_query] 开始执行 EXPLAIN，SQL=\n{sql}")

    result = _invoke_db_tool("db_explain_query", sql=sql)

    if writer:
        writer("[db_explain_query] EXPLAIN 执行完成")

    return result


@tool("db_execute_query")
def db_execute_query(
    sql: str,
    max_rows: int = 1000,
    runtime: ToolRuntime = None,
) -> dict:
    """
    执行一个只读 SQL 查询（必须是 SELECT / WITH）。
    注意：内部已经限制不能执行 DDL / DML，只能 DQL。
    """
    writer = getattr(runtime, "stream_writer", None) if runtime is not None else None
    if writer:
        writer(
            f"[db_execute_query] 准备执行只读查询（max_rows={max_rows}）：\n{sql}"
        )

    try:
        result = _invoke_db_tool("db_execute_query", sql=sql, max_rows=max_rows)
        if writer:
            writer("[db_execute_query] 查询执行完成")
        return result
    except Exception as e:
        if writer:
            writer(f"[db_execute_query] 查询执行失败：{e}")
        raise


# 收集所有工具（给 agent 用）
tools = [
    db_list_tables,
    db_get_table_schema,
    db_get_table_ddl,
    db_preview_table,
    db_explain_query,
    db_execute_query,
]


# ================== 4. NL2SQL 专用 System Prompt（必须是字符串） ==================

NL2SQL_SYSTEM_PROMPT = """
你是一个 NL2SQL 助手，负责把用户的自然语言问题转换为安全的 SQL 查询，
并使用提供的数据库工具来获取答案。

严格要求：
1. 只能生成并执行只读查询语句（SELECT 或 WITH），禁止 INSERT/UPDATE/DELETE/DDL 等。
2. 在生成 SQL 之前，优先通过工具查看表结构/字段信息（db_list_tables, db_get_table_schema, db_preview_table）。
3. 生成 SQL 后，优先调用 db_explain_query 检查 SQL 是否可执行、是否合理；
   如果 EXPLAIN 报错或不合理，需要修改 SQL 并重试，而不是直接执行。
4. 只有在确认 SQL 安全且通过 EXPLAIN 检查后，才调用 db_execute_query 获取真实数据。
5. 最终回答用户时，用中文自然语言总结查询结果，并给出使用的 SQL 语句。
6. 如果无法回答（比如表/字段不存在），要明确说明原因，不要编造结果。
"""


# ================== 5. 创建 NL2SQL Agent ==================

agent = create_agent(
    model=deepseek,
    tools=tools,               # 使用上面封装好的工具
    state_schema=CustomState,  # 自定义状态
    system_prompt=NL2SQL_SYSTEM_PROMPT,  # 这里必须是字符串
    middleware=[TodoListMiddleware(
system_prompt="你是一个严格的sql数据分析工程师，每个任务必须拆分并跟踪。",
        tool_description="""
    更新待办任务列表。todos 字段必须是字符串数组，
    且每一项代表一个可执行的任务步骤。
    """
    )],  # ✅ 加上 ToDoList 中间件

)


# 方便本地代码直接调用的封装（非流式）
def run_nl2sql(query: str, user_name: str = "default_user"):
    """
    简单封装一下 agent 的调用，传入一个自然语言问题，返回 agent 的最终回答。
    """
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": query}],
            "user_name": user_name,
        }
    )
    todos = result.get("todos")
    if todos is not None:
        print("当前 ToDo 列表:", todos)

    return result


# ================== 6. 带 Stream Writer 的测试入口（打印执行过程） ==================

if __name__ == "__main__":
    # 几条自然语言问题，测试 NL2SQL + 工具流式日志
    test_questions = [
        # "列出所有表名。",
        # "查询 orders 表中最近 5 条订单，显示 orderid、customerid、orderdate。",
        # "统计每个客户的订单数量，按数量从高到低排序，取前 10 个。",
    # "在分组汇总每个客户、每年每月购买每个商品的销售额基础上，检索每个客户哪个年月份的销售额最大。",
        "销售额最大的前30个客户中有哪些是浙江省的客户，列出这些客户的全部信息。"
    ]

    for q in test_questions:
        print("=" * 80)
        print("用户问题:", q)

        try:
            # 使用 stream，开启 updates + custom 两种流
            for stream_mode, chunk in agent.stream(
                {
                    "messages": [{"role": "user", "content": q}],
                    "user_name": "test_user",
                },
                stream_mode=["updates", "custom"],
            ):
                if stream_mode == "custom":
                    # custom 模式：就是各个 tool 里 runtime.stream_writer 写出来的内容（字符串）
                    print(f"[Tool Stream] {chunk}")
                elif stream_mode == "updates":
                    # updates 模式：每一步 agent/tool 的状态更新，这里简单打印
                    print(f"[Updates] {chunk}")

        except Exception as e:
            print("执行出错：", e)

        print()  # 每个问题之间空一行