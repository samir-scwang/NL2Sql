# main.py

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from Nl2Sql import myllm


# 引入 DeepSeek（你要的语法）
deepseek = myllm.deepseek


# --- Context 定义 ---
@dataclass
class Context:
    db_name: str


# --- 工具：执行 SQL 查询 ---
@tool
def run_sql(runtime: ToolRuntime[Context], query: str) -> str:
    """执行 SQL，演示 ToolRuntime 的使用"""

    # ToolRuntime.context
    db = runtime.context.db_name

    # ToolRuntime.stream：输出执行细节
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"[SQL Tool] Running on DB: {db}\n")
    # 假装执行 SQL
    print(db)
    print(query)
    print(runtime.state)
    print(runtime.context) #Context(db_name='user_db')
    print(runtime.config)
    print(runtime.store)
    return f"[Result from {db}] -> {query}"


# --- 创建 Agent ---
agent = create_agent(
    model=deepseek,          # DeepSeek 在这里使用
    tools=[run_sql],
    context_schema=Context
)


# --- 调用 agent ---
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "把“获取所有用户的名字”转换成 SQL 并执行。"}
        ]
    },
    context=Context(db_name="user_db")
)

last_msg = result["messages"][-1]
print(last_msg.content)
# inputs = {
#     "messages": [
#         {"role": "user", "content": "把“获取所有用户的名字”转换成 SQL 并执行。"}
#     ]
# }
#
# for chunk in agent.stream(
#     inputs,
#     context=Context(db_name="user_db"),
#     stream_mode="custom",   # 👈 关键
# ):
#     print("STREAM CHUNK:", chunk)