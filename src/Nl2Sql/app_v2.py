import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk

if __package__:
    from .agent import agent  # type: ignore
    from .chat_store import RedisChatStore, ConversationSummary  # type: ignore
else:
    _MODULE_DIR = Path(__file__).resolve().parent
    _SRC_DIR = _MODULE_DIR.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

    from Nl2Sql.agent import agent  # type: ignore
    from Nl2Sql.chat_store import RedisChatStore, ConversationSummary  # type: ignore


chat_store = RedisChatStore()
from langchain_core.messages import AIMessage, AIMessageChunk

def nl2sql_stream_generator(past_messages, user_label):
    """
    把 agent.stream 的输出包装成一个 text generator，
    给 st.write_stream 用。
    """
    last_text = ""

    for stream_mode, chunk in agent.stream(
        {
            "messages": past_messages,
            "user_name": user_label,
        },
        stream_mode=["updates", "values", "custom"],
    ):
        # 处理工具日志、todos 状态
        if stream_mode == "custom":
            text = str(chunk)
            # 这里你可以更新 log_placeholder
            # log_buffer += text + "\n"
            # log_placeholder.code(log_buffer, language="text")
            continue

        if stream_mode in {"updates", "values"} and isinstance(chunk, dict):
            # 1) 更新 todos（侧边栏）
            if "todos" in chunk:
                st.session_state["todos"] = chunk["todos"]
                # 注意：这里用 sidebar 的 placeholder
                # todo_sidebar_placeholder.markdown(...)

            # 2) 抽出模型最新消息，做流式增量
            if "model" in chunk:
                model_part = chunk["model"]
                msgs = model_part.get("messages") or []
                if not msgs:
                    continue

                last_msg = msgs[-1]
                if isinstance(last_msg, (AIMessage, AIMessageChunk)):
                    text = extract_text_from_message(last_msg)
                else:
                    text = str(last_msg)

                if not text:
                    continue

                # 做增量 diff
                if text.startswith(last_text):
                    delta = text[len(last_text):]
                else:
                    delta = text

                if delta:
                    last_text = text
                    # 把 delta 交给 st.write_stream
                    yield delta


def render_todos_md(todos: Any) -> str:
    """
    把 todos 以 HTML + Markdown 的形式渲染出来，带状态徽标。
    兼容 list[str] / list[dict] / {"todos": [...]} 等常见结构。
    """
    if not todos:
        return "<span style='color:#888;'>暂无待办任务</span>"

    candidate = todos
    if isinstance(todos, dict):
        candidate = (
            todos.get("todos")
            or todos.get("items")
            or todos.get("list")
            or todos
        )

    parsed: List[Dict[str, Any]] = []

    def normalize_status(raw: Any) -> str:
        if not raw:
            return "pending"
        lowered = str(raw).lower()
        if "complete" in lowered or lowered in {"done", "finished"}:
            return "completed"
        if "progress" in lowered or lowered in {"doing", "running"}:
            return "in_progress"
        if "todo" in lowered or "pending" in lowered:
            return "pending"
        return lowered

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
                parsed.append(
                    {
                        "text": text,
                        "status": normalize_status(
                            item.get("status") or item.get("state")
                        ),
                    }
                )
            else:
                parsed.append({"text": str(item), "status": "pending"})
    else:
        try:
            return (
                "<pre style='font-size:12px;'>"
                + json.dumps(candidate, ensure_ascii=False, indent=2, default=str)
                + "</pre>"
            )
        except Exception:
            return f"<pre>{candidate}</pre>"

    groups: Dict[str, List[Dict[str, Any]]] = {
        "in_progress": [],
        "pending": [],
        "completed": [],
        "other": [],
    }
    for item in parsed:
        status = item.get("status") or "pending"
        if status == "in_progress":
            groups["in_progress"].append(item)
        elif status == "completed":
            groups["completed"].append(item)
        elif status in {"pending", "todo", ""}:
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
        for item in items:
            status = item.get("status", "")
            text = item.get("text", "")
            if status == "completed":
                badge_color = "#16a34a"
                badge_label = "已完成"
            elif status == "in_progress":
                badge_color = "#2563eb"
                badge_label = "进行中"
            elif status in {"pending", "todo", ""}:
                badge_color = "#6b7280"
                badge_label = "待开始"
            else:
                badge_color = "#92400e"
                badge_label = status

            badge = (
                "<span style='display:inline-block;"
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

    sections = [
        render_group("进行中任务", "🔄", "#2563eb", groups["in_progress"]),
        render_group("待开始任务", "📝", "#6b7280", groups["pending"]),
        render_group("已完成任务", "✅", "#16a34a", groups["completed"]),
        render_group("其他状态", "📌", "#92400e", groups["other"]),
    ]
    html = "\n".join([section for section in sections if section])
    return html or "<span style='color:#888;'>暂无待办任务</span>"


def format_conversation_label(summary: ConversationSummary) -> str:
    title = summary.title or "未命名会话"
    if summary.updated_at:
        ts = datetime.fromtimestamp(summary.updated_at)
        ts_str = ts.strftime("%m-%d %H:%M")
    else:
        ts_str = "刚刚"
    return f"{title} · {ts_str}"


def persist_history_to_store() -> None:
    if not chat_store.is_available:
        return
    conversation_id = st.session_state.get("conversation_id")
    history = st.session_state.get("history")
    if not conversation_id or not isinstance(history, list):
        return
    chat_store.save_messages(conversation_id, history)


def extract_text_from_message(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def ensure_conversation_exists() -> Optional[str]:
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id:
        return conversation_id
    if not chat_store.is_available:
        return None
    new_id = chat_store.create_conversation()
    if new_id:
        st.session_state["conversation_id"] = new_id
        st.session_state["history"] = []
        st.session_state["loaded_conversation_id"] = new_id
    return new_id


def load_history_for_conversation(conversation_id: Optional[str]) -> None:
    if (
        not conversation_id
        or not chat_store.is_available
        or st.session_state.get("loaded_conversation_id") == conversation_id
    ):
        return
    history = chat_store.load_messages(conversation_id)
    st.session_state["history"] = history
    st.session_state["loaded_conversation_id"] = conversation_id


def delete_active_conversation() -> bool:
    conversation_id = st.session_state.get("conversation_id")
    if not conversation_id or not chat_store.is_available:
        return False
    deleted = chat_store.delete_conversation(conversation_id)
    if deleted:
        st.session_state["conversation_id"] = None
        st.session_state["loaded_conversation_id"] = None
        st.session_state["history"] = []
        st.session_state["todos"] = None
    return deleted


def trigger_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn:
        rerun_fn()


st.set_page_config(page_title="NL2SQL Demo", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] > div {
        height: 100%;
    }
    .chat-sql-sidebar {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .chat-sql-footer a {
        color: inherit;
        text-decoration: none;
    }
    .chat-sql-footer {
        margin-top: auto;
        width: 100%;
        padding-top: 32px;
        padding-bottom: 20px;
        font-size: 14px;
        line-height: 1.5;
        color: #475569;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = None
if "loaded_conversation_id" not in st.session_state:
    st.session_state["loaded_conversation_id"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []
if "todos" not in st.session_state:
    st.session_state["todos"] = None

conversation_summaries: List[ConversationSummary] = []
if chat_store.is_available:
    conversation_summaries = chat_store.list_conversations(limit=50)
    if not conversation_summaries:
        created = chat_store.create_conversation()
        if created:
            conversation_summaries = chat_store.list_conversations(limit=50)
            st.session_state["conversation_id"] = created
            st.session_state["loaded_conversation_id"] = created
            st.session_state["history"] = []
    else:
        available_ids = [s.conversation_id for s in conversation_summaries]
        if st.session_state["conversation_id"] not in available_ids:
            st.session_state["conversation_id"] = available_ids[0]

with st.sidebar:
    st.markdown("<div class='chat-sql-sidebar'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="
            margin:-4px 0 20px;
            display:flex;
            flex-direction:column;
            align-items:center;
            gap:10px;
            padding-top:4px;
            border-bottom:1px solid #e2e8f0;
        ">
            <span style="
                display:inline-flex;
                align-items:center;
                justify-content:center;
                width:40px;
                height:40px;
                border-radius:12px;
                background:linear-gradient(135deg,#6366f1,#8b5cf6);
                color:#fff;
                font-weight:800;
                font-size:40px;
            ">🤖</span>
            <div style="font-size:40px;font-weight:800;color:#0f172a;text-align:center;">ChatSQL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<h3 style='margin-bottom:4px;'>💬 历史对话</h3>", unsafe_allow_html=True)
    if chat_store.is_available:
        btn_col1, btn_col2 = st.columns([1, 1])
        if btn_col1.button("➕ 新建会话", use_container_width=True):
            created_id = chat_store.create_conversation()
            if created_id:
                st.session_state["conversation_id"] = created_id
                st.session_state["history"] = []
                st.session_state["loaded_conversation_id"] = created_id
                st.session_state["todos"] = None
                conversation_summaries = chat_store.list_conversations(limit=50)
                trigger_rerun()
        if btn_col2.button("🗑 删除当前", use_container_width=True, disabled=not st.session_state.get("conversation_id")):
            if delete_active_conversation():
                conversation_summaries = chat_store.list_conversations(limit=50)
                if conversation_summaries:
                    st.session_state["conversation_id"] = conversation_summaries[0].conversation_id
                trigger_rerun()
        if conversation_summaries:
            conv_ids = [s.conversation_id for s in conversation_summaries]
            labels = {s.conversation_id: format_conversation_label(s) for s in conversation_summaries}
            current_id = st.session_state.get("conversation_id")
            default_index = conv_ids.index(current_id) if current_id in conv_ids else 0
            selected_id = st.radio(
                "历史会话",
                conv_ids,
                index=default_index,
                format_func=lambda cid: labels.get(cid, cid),
                label_visibility="collapsed",
            )
            if selected_id != st.session_state.get("conversation_id"):
                st.session_state["conversation_id"] = selected_id
                st.session_state["loaded_conversation_id"] = None
                st.session_state["todos"] = None
                load_history_for_conversation(selected_id)
                trigger_rerun()
        else:
            st.caption("暂无历史记录，点击“新建会话”开始聊天。")
    else:
        st.info("Redis 未连接，聊天历史仅保存在当前浏览器会话中。")

    st.markdown("<hr />", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-bottom:4px;'>🧩 当前任务拆解</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:12px;color:#6b7280;margin-top:0;'>自动根据你的问题拆分 SQL 分析步骤。</p>",
        unsafe_allow_html=True,
    )
    todo_sidebar_placeholder = st.empty()
    todo_sidebar_placeholder.markdown(
        render_todos_md(st.session_state["todos"]),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="chat-sql-footer">
            made by <strong>Sicheng Wang</strong><br/>
            email: <a href="mailto:samircb20619@gmail.com">samircb20619@gmail.com</a><br/>
            wechat: Yi_77ss<br/>
            Hangzhou · China
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

load_history_for_conversation(st.session_state.get("conversation_id"))

show_welcome_message = not st.session_state["history"]

if show_welcome_message:
    st.title("🧠 NL2SQL 智能查询助手")
    st.caption("自然语言 → 安全 SQL → 数据库查询（LangChain + LangGraph + 自定义 DbTools）")
    st.markdown(
        """
        <div style="
            margin: 120px auto 60px;
            max-width: 700px;
            text-align: center;
            font-size: 60px;
            line-height: 1.5;
            font-weight: 700;
            color: #0f172a;
        ">
            让我们开始编写 SQL 吧！
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("请输入自然语言问题，例如：销售额最大的前10个客户中哪些来自浙江？")

if user_input:
    conversation_id = ensure_conversation_exists()
    st.session_state["history"].append({"role": "user", "content": user_input})
    persist_history_to_store()

    with st.chat_message("user"):
        st.markdown(user_input)

    assistant_block = st.chat_message("assistant")
    answer_placeholder = assistant_block.empty()
    thinking_expander = st.expander("🧠 深度思考（模型中间回复）", expanded=False)
    thinking_placeholder = thinking_expander.empty()

    log_expander = st.expander("🛠️ 工具执行过程（ToolRuntime Stream）", expanded=False)
    updates_expander = st.expander("🧩 Agent 内部状态更新（调试用）", expanded=False)
    log_placeholder = log_expander.empty()
    updates_placeholder = updates_expander.empty()

    log_buffer = ""
    updates_buffer = ""
    final_answer = ""
    thinking_chunks: List[str] = []
    thinking_state = {"last": ""}
    assistant_state = {"last": ""}

    def update_thinking_panel(new_text: str) -> None:
        if not isinstance(new_text, str):
            return
        text = new_text.strip()
        if not text:
            return
        last_snapshot = thinking_state["last"]
        if last_snapshot and text.startswith(last_snapshot):
            incremental = text[len(last_snapshot):]
        else:
            incremental = text
        incremental = incremental.strip("\n")
        if incremental:
            thinking_chunks.append(incremental)
            thinking_placeholder.markdown("\n\n".join(thinking_chunks))
        thinking_state["last"] = text

    def update_answer_panel(new_text: str) -> None:
        if not isinstance(new_text, str):
            return
        assistant_state["last"] = new_text
        answer_placeholder.markdown(new_text or " ")

    past_messages = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state["history"]
    ]
    user_label = f"web_user_{conversation_id}" if conversation_id else "web_user"

    try:
        for stream_mode, chunk in agent.stream(
            {
                "messages": past_messages,
                "user_name": user_label,
            },
            stream_mode=["updates", "custom", "values"],
        ):
            if stream_mode == "custom":
                text = str(chunk)
                log_buffer += text + "\n"
                log_placeholder.code(log_buffer, language="text")
            elif stream_mode == "updates":
                try:
                    pretty = json.dumps(chunk, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    pretty = str(chunk)
                updates_buffer += pretty + "\n" + "-" * 60 + "\n"
                updates_placeholder.code(updates_buffer, language="json")

                if isinstance(chunk, dict) and "model" in chunk:
                    model_part: Dict[str, Any] = chunk["model"]  # type: ignore
                    msgs = model_part.get("messages") or []
                    if isinstance(msgs, list) and msgs:
                        last_msg = msgs[-1]
                        if isinstance(last_msg, (AIMessage, AIMessageChunk)):
                            msg_text = extract_text_from_message(last_msg)
                            if msg_text:
                                final_answer = msg_text
                                update_answer_panel(msg_text)
                                update_thinking_panel(msg_text)

                if isinstance(chunk, dict) and "todos" in chunk:
                    st.session_state["todos"] = chunk["todos"]
                    todo_sidebar_placeholder.markdown(
                        render_todos_md(chunk["todos"]),
                        unsafe_allow_html=True,
                    )
            elif stream_mode == "values":
                if isinstance(chunk, dict):
                    if "model" in chunk:
                        model_part = chunk["model"]  # type: ignore
                        msgs = model_part.get("messages") or []
                        if isinstance(msgs, list) and msgs:
                            last_msg = msgs[-1]
                            if isinstance(last_msg, (AIMessage, AIMessageChunk)):
                                msg_text = extract_text_from_message(last_msg)
                                if msg_text:
                                    final_answer = msg_text
                                    update_answer_panel(msg_text)
                                    update_thinking_panel(msg_text)
                    if "todos" in chunk:
                        st.session_state["todos"] = chunk["todos"]
                        todo_sidebar_placeholder.markdown(
                            render_todos_md(chunk["todos"]),
                            unsafe_allow_html=True,
                        )

        if not final_answer:
            final_answer = assistant_state["last"]

        if final_answer:
            answer_placeholder.markdown(final_answer)
            st.session_state["history"].append({"role": "assistant", "content": final_answer})
        else:
            fallback = "（执行完成，但没有解析到最终回答，请查看工具日志。）"
            answer_placeholder.markdown(fallback)
            st.session_state["history"].append({"role": "assistant", "content": fallback})
        persist_history_to_store()
    except Exception as exc:
        err_text = f"⚠️ 调用 agent 失败：{exc}"
        answer_placeholder.error(err_text)
        st.session_state["history"].append({"role": "assistant", "content": err_text})
        persist_history_to_store()