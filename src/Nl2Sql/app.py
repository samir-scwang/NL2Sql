
import sys
from pathlib import Path
import json
from typing import Dict, Any, List

import streamlit as st
from langchain_core.messages import AIMessage

# ========= 兼容导入：从 agent.py 引入 agent 对象 =========

if __package__:
    # 作为包导入：from Nl2Sql import app
    from .agent import agent  # type: ignore
else:
    # 直接执行：streamlit run src/Nl2Sql/app.py
    _MODULE_DIR = Path(__file__).resolve().parent
    _SRC_DIR = _MODULE_DIR.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

    from Nl2Sql.agent import agent  # type: ignore


# ========= 辅助函数：渲染 ToDoList（美化版） =========

def render_todos_md(todos: Any) -> str:
    """
    把 todos 以 HTML+Markdown 的形式渲染出来，带状态徽标：
      - completed: 绿色 ✅
      - in_progress: 蓝色 🔄
      - pending / todo: 灰色 ⏳
    尽量兼容几种结构：
      - list[str]
      - {"todos": [...]} / {"items": [...]} / {"list": [...]}
      - list[dict]，dict 里有 content/task/description/title/status 等字段
    """
    if not todos:
        return "<span style='color:#888;'>暂无待办任务</span>"

    # 尝试从字典中拿到真正的列表
    candidate = todos
    if isinstance(todos, dict):
        candidate = (
            todos.get("todos")
            or todos.get("items")
            or todos.get("list")
            or todos
        )

    # 统一解析成：[{text, status}]
    parsed: List[Dict[str, Any]] = []

    def normalize_status(raw: Any) -> str:
        if not raw:
            return "pending"
        s = str(raw).lower()
        if "complete" in s or s in {"done", "finished"}:
            return "completed"
        if "progress" in s or "doing" in s or s in {"running", "executing"}:
            return "in_progress"
        if "todo" in s or "pending" in s or "plan" in s:
            return "pending"
        return s  # 其他状态原样返回

    if isinstance(candidate, list):
        for item in candidate:
            if isinstance(item, str):
                parsed.append({"text": item, "status": "pending"})
            elif isinstance(item, dict):
                text = (
                    item.get("content")
                    or item.get("task")
                    or item.get("description")
                    or item.get("title")
                    or str(item)
                )
                status = normalize_status(item.get("status") or item.get("state"))
                parsed.append({"text": text, "status": status})
            else:
                parsed.append({"text": str(item), "status": "pending"})
    else:
        # 奇怪结构，直接原样 JSON 输出
        try:
            return (
                "<pre style='font-size:12px;'>"
                + json.dumps(todos, ensure_ascii=False, indent=2, default=str)
                + "</pre>"
            )
        except Exception:
            return f"<pre>{str(todos)}</pre>"

    # 分组：进行中 / 待开始 / 已完成 / 其他
    groups: Dict[str, List[Dict[str, Any]]] = {
        "in_progress": [],
        "pending": [],
        "completed": [],
        "other": [],
    }

    for item in parsed:
        status = item["status"]
        if status == "in_progress":
            groups["in_progress"].append(item)
        elif status == "completed":
            groups["completed"].append(item)
        elif status in ("pending", "todo", None, ""):
            groups["pending"].append(item)
        else:
            groups["other"].append(item)

    def render_group(title: str, emoji: str, color: str, items: List[Dict[str, Any]]) -> str:
        if not items:
            return ""
        html = [
            f"<div style='margin-top:4px;margin-bottom:2px;font-weight:600;color:{color};font-size:13px;'>{emoji} {title}</div>",
            "<ul style='padding-left:18px;margin-top:0;margin-bottom:4px;'>",
        ]
        for it in items:
            status = it.get("status", "pending")
            text = it.get("text", "")
            # 状态徽章
            if status == "completed":
                badge_color = "#16a34a"  # 绿
                badge_label = "已完成"
            elif status == "in_progress":
                badge_color = "#2563eb"  # 蓝
                badge_label = "进行中"
            elif status in ("pending", "todo", None, ""):
                badge_color = "#6b7280"  # 灰
                badge_label = "待开始"
            else:
                badge_color = "#92400e"  # 棕/其他
                badge_label = status

            badge = (
                f"<span style='display:inline-block;"
                f"padding:1px 6px;margin-right:6px;border-radius:999px;"
                f"font-size:11px;background-color:{badge_color}20;"
                f"color:{badge_color};border:1px solid {badge_color}40;'>"
                f"{badge_label}</span>"
            )

            html.append(
                "<li style='margin-bottom:2px;font-size:13px;line-height:1.4;'>"
                f"{badge}{text}</li>"
            )
        html.append("</ul>")
        return "\n".join(html)

    parts: List[str] = ["<div style='font-size:13px;'>"]

    parts.append(render_group("进行中任务", "🔄", "#2563eb", groups["in_progress"]))
    parts.append(render_group("待开始任务", "⏳", "#6b7280", groups["pending"]))
    parts.append(render_group("已完成任务", "✅", "#16a34a", groups["completed"]))
    parts.append(render_group("其他状态", "📌", "#92400e", groups["other"]))

    parts.append("</div>")

    html = "\n".join([p for p in parts if p])  # 去掉空段落
    if not html.strip():
        html = "<span style='color:#888;'>暂无待办任务</span>"

    return html


# ========= Streamlit 页面基础设置 =========

st.set_page_config(
    page_title="NL2SQL Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NL2SQL 智能查询助手")
st.caption("自然语言 → 安全 SQL → 数据库查询（LangChain + LangGraph + 自定义 DbTools）")

# ========= 会话状态初始化 =========

if "history" not in st.session_state:
    # 每条记录：{"role": "user" | "assistant", "content": str}
    st.session_state["history"] = []

if "todos" not in st.session_state:
    # 当前 ToDoList（由 TodoListMiddleware 维护，这里只负责展示）
    st.session_state["todos"] = None

# ========= 侧边栏：ToDoList 面板 =========

with st.sidebar:
    st.markdown(
        "<h3 style='margin-bottom:4px;'>✅ 当前任务拆解</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:12px;color:#6b7280;margin-top:0;'>自动根据你的问题拆分 SQL 分析步骤。</p>",
        unsafe_allow_html=True,
    )
    todo_sidebar_placeholder = st.empty()
    todo_sidebar_placeholder.markdown(
        render_todos_md(st.session_state["todos"]),
        unsafe_allow_html=True,   # ✅ 允许 HTML 渲染徽章 / 颜色
    )

# ========= 历史对话展示 =========

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ========= 输入框 =========

user_input = st.chat_input("请输入你的自然语言问题，比如：销售额最大的前30个客户中有哪些是浙江省的客户？")

# ========= 处理输入 =========

if user_input:
    # 1. 把用户问题加入历史 & 展示
    st.session_state["history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. 准备助手回复区域（一个 message 里，内容不断刷新）
    assistant_block = st.chat_message("assistant")
    answer_placeholder = assistant_block.empty()

    # 3. 准备两个折叠区域：一个实时显示工具执行日志，一个显示 updates（调试用）
    log_expander = st.expander("🛠️ 工具执行过程（ToolRuntime Stream）", expanded=False)
    updates_expander = st.expander("🧩 Agent 内部状态更新（调试用）", expanded=False)

    log_placeholder = log_expander.empty()
    updates_placeholder = updates_expander.empty()

    # 4. 缓冲区
    log_buffer = ""
    updates_buffer = ""
    final_answer = ""   # 用于记录“最新的 AI 回复”

    try:
        # 使用 stream：同时监听 updates / custom / values
        for stream_mode, chunk in agent.stream(
            {
                "messages": [{"role": "user", "content": user_input}],
                "user_name": "web_user",
            },
            stream_mode=["updates", "custom", "values"],
        ):
            # ---------- custom：ToolRuntime.stream_writer 输出 ----------
            if stream_mode == "custom":
                text = str(chunk)
                log_buffer += text + "\n"
                log_placeholder.code(log_buffer, language="text")

            # ---------- updates：LangGraph 状态更新（调试用） ----------
            elif stream_mode == "updates":
                try:
                    pretty = json.dumps(chunk, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    pretty = str(chunk)

                updates_buffer += pretty + "\n" + "-" * 60 + "\n"
                updates_placeholder.code(updates_buffer, language="json")

                # 从 updates 中提取当前模型回复
                if isinstance(chunk, dict) and "model" in chunk:
                    model_part: Dict[str, Any] = chunk["model"]  # type: ignore
                    msgs = model_part.get("messages") or []
                    if isinstance(msgs, list) and msgs:
                        last_msg = msgs[-1]
                        if isinstance(last_msg, AIMessage):
                            if isinstance(last_msg.content, str):
                                final_answer = last_msg.content
                                answer_placeholder.markdown(final_answer)

                # 有些版本也会在 updates 里带上 todos，这里也顺带处理一下
                if isinstance(chunk, dict) and "todos" in chunk:
                    new_todos = chunk["todos"]
                    st.session_state["todos"] = new_todos
                    todo_sidebar_placeholder.markdown(
                        render_todos_md(new_todos),
                        unsafe_allow_html=True,
                    )

            # ---------- values：最终 state（包含完整 todos 等） ----------
            elif stream_mode == "values":
                if isinstance(chunk, dict):
                    # 1）最终回答（有些实现会直接在 values 里给完整 messages）
                    if "model" in chunk:
                        model_part: Dict[str, Any] = chunk["model"]  # type: ignore
                        msgs = model_part.get("messages") or []
                        if isinstance(msgs, list) and msgs:
                            last_msg = msgs[-1]
                            if isinstance(last_msg, AIMessage) and isinstance(last_msg.content, str):
                                final_answer = last_msg.content
                                answer_placeholder.markdown(final_answer)

                    # 2）最终 ToDoList
                    if "todos" in chunk:
                        new_todos = chunk["todos"]
                        st.session_state["todos"] = new_todos
                        todo_sidebar_placeholder.markdown(
                            render_todos_md(new_todos),
                            unsafe_allow_html=True,
                        )

        # ========== 流结束：把最终答案写入历史 ==========
        if final_answer:
            answer_placeholder.markdown(final_answer)
            st.session_state["history"].append(
                {"role": "assistant", "content": final_answer}
            )
        else:
            fallback = "（执行已完成，但没有解析到最终回答内容，请查看上面的执行日志和状态更新。）"
            answer_placeholder.markdown(fallback)
            st.session_state["history"].append(
                {"role": "assistant", "content": fallback}
            )

    except Exception as e:
        err_text = f"❌ 调用 agent 失败：{e}"
        answer_placeholder.error(err_text)
        st.session_state["history"].append(
            {"role": "assistant", "content": err_text}
        )
