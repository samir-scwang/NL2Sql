import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk

# -----------------------------------------------------------------------------
# 1. 环境与路径配置
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 2. Page Config & Premium UI CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="NL2SQL Pro", page_icon="✨", layout="wide")


def get_premium_style_css(is_dark: bool = False):
    if is_dark:
        # --- Dark Mode (Deep Space & Neon) ---
        c_bg = "#0f172a"         # Slate 900
        c_bg_sec = "#1e293b"     # Slate 800
        c_text = "#f1f5f9"       # Slate 100
        c_text_muted = "#94a3b8" # Slate 400
        c_primary = "#818cf8"    # Indigo 400
        c_primary_hover = "#6366f1" # Indigo 500
        
        # Glassmorphism
        c_glass_bg = "rgba(30, 41, 59, 0.7)"
        c_glass_border = "rgba(255, 255, 255, 0.08)"
        c_glass_shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.3)"
        
        # Gradients
        c_grad_text = "linear-gradient(135deg, #818cf8 0%, #c084fc 100%)"
        c_grad_bg = "linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)"
        
        # Inputs
        c_input_bg = "rgba(30, 41, 59, 0.6)"
        c_input_border = "rgba(255, 255, 255, 0.1)"
    else:
        # --- Light Mode (Frost & Clean) ---
        c_bg = "#f8fafc"         # Slate 50
        c_bg_sec = "#ffffff"     # White
        c_text = "#1e293b"       # Slate 800
        c_text_muted = "#64748b" # Slate 500
        c_primary = "#4f46e5"    # Indigo 600
        c_primary_hover = "#4338ca" # Indigo 700
        
        # Glassmorphism
        c_glass_bg = "rgba(255, 255, 255, 0.7)"
        c_glass_border = "rgba(255, 255, 255, 0.4)"
        c_glass_shadow = "0 8px 32px 0 rgba(31, 38, 135, 0.07)"
        
        # Gradients
        c_grad_text = "linear-gradient(135deg, #4f46e5 0%, #9333ea 100%)"
        c_grad_bg = "linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%)"
        
        # Inputs
        c_input_bg = "rgba(255, 255, 255, 0.8)"
        c_input_border = "rgba(226, 232, 240, 0.8)"

    return f"""
    <style>
        :root {{
            --bg-secondary: {c_bg_sec};
            --text-color: {c_text};
            --glass-border: {c_glass_border};
            --primary-color: {c_primary};
        }}

        /* --- Fonts & Reset --- */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: {c_text};
            background-color: {c_bg};
            scroll-behavior: smooth;
        }}
        div[data-testid="stSidebarHeader"] {{
    height: 10px !important;    
    min-height: 20px !important;
    padding: 0 !important;
    margin: 0 !important;
}}
        /* --- Animations --- */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes pulse-glow {{
            0% {{ box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }}
            70% {{ box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }}
        }}
        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-6px); }}
            100% {{ transform: translateY(0px); }}
        }}

        /* 思考中动画 - 跳动圆点 */
        @keyframes thinkingBounce {{
            0% {{ transform: translateY(0); opacity: 0.4; }}
            50% {{ transform: translateY(-4px); opacity: 1; }}
            100% {{ transform: translateY(0); opacity: 0.4; }}
        }}
        .thinking-wrapper {{
            display: inline-flex;
            flex-direction: column;
            gap: 6px;
        }}
        .thinking-label {{
            font-size: 13px;
            color: {c_text_muted};
        }}
        .thinking-dots {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .thinking-dot {{
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: {c_primary};
            animation: thinkingBounce 1s infinite;
        }}
        .thinking-dot:nth-child(2) {{
            animation-delay: 0.15s;
        }}
        .thinking-dot:nth-child(3) {{
            animation-delay: 0.3s;
        }}

        /* --- Layout Cleanup --- */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{
            background: transparent;
            backdrop-filter: blur(0px);
            height: 0px;
        }}
        
        /* --- Sidebar (Glassmorphism) --- */
        section[data-testid="stSidebar"] {{
            background-color: {c_bg_sec};
            border-right: 1px solid {c_glass_border};
            box-shadow: 4px 0 24px rgba(0,0,0,0.02);
        }}
        section[data-testid="stSidebar"] > div {{
            padding: 30px 20px;
        }}
        
        /* Sidebar Nav Items */
        .sidebar-label {{
            font-size: 11px;
            font-weight: 800;
            color: {c_text_muted};
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin: 24px 0 12px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .sidebar-label::after {{
            content: "";
            flex: 1;
            height: 1px;
            background: {c_glass_border};
            opacity: 0.5;
        }}

        /* --- Buttons (Modern Pill) --- */
        div.stButton > button {{
            width: 100%;
            background: {c_bg_sec};
            border: 1px solid {c_glass_border};
            color: {c_text};
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
            padding: 0.6rem 1rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }}
        div.stButton > button:hover {{
            border-color: {c_primary};
            color: {c_primary};
            background: {c_glass_bg};
            transform: translateY(-2px);
            box-shadow: 0 8px 16px -4px rgba(99, 102, 241, 0.15);
        }}
        div.stButton > button:active {{
            transform: translateY(0);
        }}

        /* --- Radio Group (Nav Style) --- */
        div[role="radiogroup"] label {{
            padding: 10px 16px !important;
            border-radius: 12px;
            border: 1px solid transparent;
            transition: all 0.2s;
            margin-bottom: 4px;
            background: transparent;
        }}
        div[role="radiogroup"] label:hover {{
            background: {c_glass_bg};
            border-color: {c_glass_border};
        }}
        div[role="radiogroup"] [data-checked="true"] {{
            background: {c_glass_bg} !important;
            border: 1px solid {c_primary} !important;
        }}
        div[role="radiogroup"] [data-checked="true"] + div {{
            color: {c_primary} !important;
            font-weight: 700 !important;
        }}

        /* --- Chat Interface --- */
        .stChatInputContainer {{
            padding-bottom: 40px;
            background: linear-gradient(to top, {c_bg} 90%, transparent 100%);
        }}
        div[data-testid="stChatInput"] {{
            border-radius: 24px !important;
            border: 1px solid {c_input_border} !important;
            background: {c_input_bg} !important;
            backdrop-filter: blur(12px);
            box-shadow: {c_glass_shadow} !important;
            padding: 2px;
            transition: all 0.3s ease;
        }}
        div[data-testid="stChatInput"]:focus-within {{
            border-color: {c_primary} !important;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2), {c_glass_shadow} !important;
            transform: translateY(-2px);
        }}
        div[data-testid="stChatInput"] textarea {{
            color: {c_text};
            font-weight: 500;
        }}

        /* Messages */
        div[data-testid="stChatMessage"] {{
            background: transparent;
            padding: 0rem 0;
            animation: fadeInUp 0.5s ease-out;
        }}
        
        /* User Bubble (Gradient) */
        div[data-testid="stChatMessage"][data-testid="user"] div[data-testid="stMarkdownContainer"] {{
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            padding: 14px 24px;
            border-radius: 24px 24px 4px 24px;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
            font-size: 15px;
            line-height: 1.6;
            font-weight: 500;
        }}
        
        /* Assistant Bubble (Glass Card) */
        div[data-testid="stChatMessage"][data-testid="assistant"] div[data-testid="stMarkdownContainer"] {{
            background: {c_bg_sec};
            border: 1px solid {c_glass_border};
            color: {c_text};
            padding: 18px 28px;
            border-radius: 4px 24px 24px 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            font-size: 15px;
            line-height: 1.7;
        }}

        /* --- Custom Classes --- */
        .welcome-container {{
            animation: fadeInUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
        }}
        .welcome-title {{
            font-size: 64px;
            font-weight: 800;
            letter-spacing: -2px;
            line-height: 1.1;
            margin-bottom: 20px;
            background: {c_grad_text};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .welcome-subtitle {{
            font-size: 20px;
            color: {c_text_muted};
            font-weight: 500;
            max-width: 600px;
            margin: 0 auto 48px;
            line-height: 1.6;
        }}
        .welcome-card {{
            padding: 12px 24px;
            background: {c_bg_sec};
            border: 1px solid {c_glass_border};
            border-radius: 100px;
            color: {c_text};
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .welcome-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
            border-color: {c_primary};
            color: {c_primary};
        }}
        
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
        
        .chat-footer {{
            margin-top: auto;
            padding: 32px 0;
            text-align: center;
            border-top: 1px solid {c_glass_border};
        }}
        .chat-footer-text {{
            font-size: 12px;
            color: {c_text_muted};
            font-weight: 500;
        }}
    </style>
    """


# -----------------------------------------------------------------------------
# 3. UI 渲染函数 (组件化设计)
# -----------------------------------------------------------------------------

def render_sidebar_header():
    st.markdown(
        """<div style="display:flex; align-items:center; gap:16px; padding-bottom:30px;">
            <div style="
                width:48px; height:48px; border-radius:14px; 
                background: linear-gradient(135deg, #6366f1, #a855f7);
                display:flex; align-items:center; justify-content:center;
                color:white; font-size:26px; 
                box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
                animation: float 6s ease-in-out infinite;
            ">
                ✨
            </div>
            <div>
                <div style="font-family:'Plus Jakarta Sans'; font-weight:800; font-size:22px; color:var(--text-color); letter-spacing:-0.5px;">NextSQL</div>
                <div style="font-size:12px; color:#94a3b8; font-weight:600; letter-spacing:0.5px;">ENTERPRISE AGENT</div>
            </div>
        </div>""",
        unsafe_allow_html=True
    )


def render_todos_premium(todos: Any) -> str:
    """渲染高级卡片式任务列表 (Glassmorphism + Grouping)"""
    if not todos:
        return """<div style='text-align:center; padding:40px 20px; background: rgba(255,255,255,0.05); border-radius:16px; border:1px dashed rgba(148, 163, 184, 0.4);'>
            <div style='font-size:24px; margin-bottom:8px; opacity:0.6;'>📝</div>
            <div style='font-size:13px; color:#94a3b8; font-weight:500;'>Waiting for tasks...</div>
        </div>"""

    candidate = todos
    if isinstance(todos, dict):
        candidate = todos.get("todos") or todos.get("items") or todos.get("list") or todos

    parsed = []
    if isinstance(candidate, list):
        for item in candidate:
            if isinstance(item, dict):
                parsed.append({
                    "text": item.get("content") or item.get("task") or str(item),
                    "status": str(item.get("status") or "pending").lower()
                })
            else:
                parsed.append({"text": str(item), "status": "pending"})
    
    # Add numbering
    for i, p in enumerate(parsed, 1):
        p['index'] = i

    groups = {
        "in_progress": [],
        "pending": [],
        "completed": [],
        "other": []
    }
    
    for item in parsed:
        s = item['status']
        if "progress" in s or s in ["doing", "running"]:
            groups["in_progress"].append(item)
        elif "complete" in s or s in ["done", "finished"]:
            groups["completed"].append(item)
        elif s in ["pending", "todo", ""]:
            groups["pending"].append(item)
        else:
            groups["other"].append(item)

    def render_group(title, items, color_theme):
        if not items:
            return ""
        
        colors = {
            "blue":  {"bg": "rgba(59, 130, 246, 0.1)", "border": "rgba(59, 130, 246, 0.2)", "text": "#3b82f6", "dot": "#3b82f6"},
            "gray":  {"bg": "rgba(148, 163, 184, 0.1)", "border": "rgba(148, 163, 184, 0.2)", "text": "#94a3b8", "dot": "#94a3b8"},
            "green": {"bg": "rgba(16, 185, 129, 0.1)", "border": "rgba(16, 185, 129, 0.2)", "text": "#10b981", "dot": "#10b981"},
            "amber": {"bg": "rgba(245, 158, 11, 0.1)",  "border": "rgba(245, 158, 11, 0.2)",  "text": "#f59e0b", "dot": "#f59e0b"},
        }
        c = colors.get(color_theme, colors["gray"])
        
        html = [f'<div style="margin-top:16px; margin-bottom:8px; font-size:12px; font-weight:700; color:{c["text"]}; text-transform:uppercase; letter-spacing:1px;">{title}</div>']
        
        for item in items:
            card = f"""<div style="margin-bottom:8px; padding:12px 16px; background: var(--bg-secondary); border: 1px solid {c["border"]}; border-left: 3px solid {c["dot"]}; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); transition: all 0.2s;">
                <div style="font-size:13px; line-height:1.5; color:var(--text-color); font-weight:500;">
                    <span style="opacity:0.5; margin-right:6px;">{item['index']}.</span>{item['text']}
                </div>
            </div>"""
            html.append(card)
        return "".join(html)

    sections = [
        render_group("In Progress", groups["in_progress"], "blue"),
        render_group("Pending", groups["pending"], "gray"),
        render_group("Completed", groups["completed"], "green"),
        render_group("Other", groups["other"], "amber"),
    ]
    
    final_html = "".join(sections)
    return final_html if final_html else """<div style='text-align:center; padding:20px; color:#94a3b8; font-size:13px;'>No active tasks</div>"""


def render_welcome_screen():
    st.markdown(
        """
<div class="welcome-container" style="margin: 80px auto 60px; max-width: 900px; text-align: center;">
    <div class="welcome-title">
        Unlock Data Insights<br/>with Natural Language.
    </div>
    <div class="welcome-subtitle">
        Your intelligent data companion. Ask questions in plain English and get instant, secure SQL queries and visualizations.
    </div>
    <div style="display:flex; gap:16px; justify-content:center; flex-wrap:wrap; margin-top:40px;">
        <div class="welcome-card">📈 Sales Trend Analysis</div>
        <div class="welcome-card">🗺️ Customer Geography</div>
        <div class="welcome-card">⚠️ Inventory Alerts</div>
        <div class="welcome-card">📊 Q3 Performance Report</div>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )


def format_conversation_label(summary: ConversationSummary) -> str:
    title = summary.title or "New Conversation"
    if len(title) > 18:
        title = title[:18] + "..."
    ts = datetime.fromtimestamp(summary.updated_at).strftime("%H:%M") if summary.updated_at else ""
    return f"{title}  ·  {ts}"


def persist_history_to_store() -> None:
    if not chat_store.is_available:
        return
    conversation_id = st.session_state.get("conversation_id")
    history = st.session_state.get("history")
    if conversation_id and isinstance(history, list):
        chat_store.save_messages(conversation_id, history)


def extract_text_from_message(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join([str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content])
    return str(content)


def ensure_conversation_exists() -> Optional[str]:
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id:
        return conversation_id
    if not chat_store.is_available:
        return None
    new_id = chat_store.create_conversation()
    if new_id:
        st.session_state.update({"conversation_id": new_id, "history": [], "loaded_conversation_id": new_id})
    return new_id


def load_history_for_conversation(conversation_id: Optional[str]) -> None:
    if not conversation_id or not chat_store.is_available or st.session_state.get(
        "loaded_conversation_id") == conversation_id:
        return
    st.session_state["history"] = chat_store.load_messages(conversation_id)
    st.session_state["loaded_conversation_id"] = conversation_id


def delete_active_conversation() -> bool:
    cid = st.session_state.get("conversation_id")
    if not cid or not chat_store.is_available:
        return False
    if chat_store.delete_conversation(cid):
        st.session_state.update({
            "conversation_id": None,
            "loaded_conversation_id": None,
            "history": [],
            "todos": None,
            "welcome_dismissed": False,  # 删除后再显示欢迎页
        })
        return True
    return False


def trigger_rerun():
    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None))()


# -----------------------------------------------------------------------------
# 4. Session Init
# -----------------------------------------------------------------------------
if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []
if "todos" not in st.session_state:
    st.session_state["todos"] = None
if "welcome_dismissed" not in st.session_state:
    st.session_state["welcome_dismissed"] = False

# -----------------------------------------------------------------------------
# 5. Sidebar Layout
# -----------------------------------------------------------------------------
conversation_summaries = chat_store.list_conversations(limit=50) if chat_store.is_available else []
if chat_store.is_available and not conversation_summaries:
    created = chat_store.create_conversation()
    if created:
        conversation_summaries = chat_store.list_conversations()
        st.session_state["conversation_id"] = created

with st.sidebar:
    render_sidebar_header()
    
    # --- Theme Toggle ---
    is_dark = st.toggle("🌙 Dark Mode", value=True)
    st.markdown(get_premium_style_css(is_dark), unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-label">History</div>', unsafe_allow_html=True)
    if chat_store.is_available:
        c1, c2 = st.columns(2)
        if c1.button("＋ New Chat"):
            new_id = chat_store.create_conversation()
            if new_id:
                st.session_state.update(
                    {
                        "conversation_id": new_id,
                        "history": [],
                        "todos": None,
                        "loaded_conversation_id": new_id,
                        "welcome_dismissed": False,  # 新对话重新显示欢迎页
                    }
                )
                trigger_rerun()
        if c2.button("🗑 Delete", disabled=not st.session_state.get("conversation_id")):
            if delete_active_conversation():
                trigger_rerun()

        if conversation_summaries:
            conv_ids = [s.conversation_id for s in conversation_summaries]
            labels = {s.conversation_id: format_conversation_label(s) for s in conversation_summaries}
            curr_id = st.session_state.get("conversation_id")
            sel_id = st.radio(
                "History List", conv_ids,
                index=conv_ids.index(curr_id) if curr_id in conv_ids else 0,
                format_func=lambda x: labels.get(x, x), label_visibility="collapsed"
            )
            if sel_id != curr_id:
                st.session_state["conversation_id"] = sel_id
                load_history_for_conversation(sel_id)
                st.session_state["todos"] = None
                st.session_state["welcome_dismissed"] = True  # 切换已有对话就不显示欢迎页
                trigger_rerun()
    else:
        st.warning("Redis Disconnected")

    st.markdown('<div class="sidebar-label">Agent Tasks</div>', unsafe_allow_html=True)
    todo_ph = st.empty()
    todo_ph.markdown(render_todos_premium(st.session_state["todos"]), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="chat-footer">
            NextSQL AI © 2024<br/>
            Designed by <a href="#">Sicheng Wang</a>
        </div>
        """, unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 6. Main Interface
# -----------------------------------------------------------------------------
load_history_for_conversation(st.session_state.get("conversation_id"))

# 欢迎区域用占位符，便于之后动态清空
welcome_ph = st.empty()
if (
    not st.session_state["history"]
    and not st.session_state.get("welcome_dismissed", False)
):
    with welcome_ph:
        render_welcome_screen()

# 历史对话渲染
for msg in st.session_state["history"]:
    role = msg["role"]
    avatar = "👤" if role == "user" else "✨"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Ask anything about your data...")

if user_input:
    # 一旦用户输入第一条消息，欢迎区立刻消失
    st.session_state["welcome_dismissed"] = True
    welcome_ph.empty()

    cid = ensure_conversation_exists()
    st.session_state["history"].append({"role": "user", "content": user_input})
    persist_history_to_store()

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Assistant 区域：先展示“思考中”动画
    with st.chat_message("assistant", avatar="✨"):
        ans_ph = st.empty()
        with st.expander("🧠 Thinking Process"):
            think_ph = st.empty()
        with st.expander("🔧 Tool Logs"):
            log_ph = st.empty()

        # 初始思考中动画
        ans_ph.markdown(
            """
            <div class="thinking-wrapper">
                <div class="thinking-label">思考中...</div>
                <div class="thinking-dots">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    final_ans = ""
    log_buf = ""
    think_buf: List[str] = []

    past_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state["history"]]
    u_label = f"user_{cid}" if cid else "user"

    start_time = datetime.now()

    try:
        for mode, chunk in agent.stream(
                {"messages": past_msgs, "user_name": u_label},
                stream_mode=["updates", "custom", "values"]
        ):
            if mode == "custom":
                log_buf += str(chunk) + "\n"
                log_ph.code(log_buf, language="text")

            elif mode == "updates":
                if isinstance(chunk, dict):
                    if "model" in chunk:
                        msgs = chunk["model"].get("messages", [])
                        if msgs:
                            txt = extract_text_from_message(msgs[-1])
                            if txt:
                                final_ans = txt
                                ans_ph.markdown(txt)
                                if txt not in think_buf:
                                    think_buf.append(txt)
                                    think_ph.markdown("\n\n".join(think_buf))

                    todos = None
                    if "todos" in chunk:
                        todos = chunk["todos"]
                    elif "tools" in chunk and isinstance(chunk["tools"], dict):
                        if "todos" in chunk["tools"]:
                            todos = chunk["tools"]["todos"]

                    if todos is not None:
                        st.session_state["todos"] = todos
                        todo_ph.markdown(
                            render_todos_premium(todos),
                            unsafe_allow_html=True
                        )

            elif mode == "values":
                if isinstance(chunk, dict) and "model" in chunk:
                    msgs = chunk["model"].get("messages", [])
                    if msgs:
                        txt = extract_text_from_message(msgs[-1])
                        if txt:
                            final_ans = txt
                            ans_ph.markdown(txt)

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        if not final_ans:
            final_ans = "Process completed. Check logs for details."

        timing_note = f"\n\n---\n<span style='font-size:12px; color:#94a3b8;'>⏱ 思考用时：{elapsed:.2f} 秒</span>"
        ans_ph.markdown(final_ans + timing_note, unsafe_allow_html=True)

        st.session_state["history"].append(
            {
                "role": "assistant",
                "content": final_ans + f"\n\n(思考用时：{elapsed:.2f} 秒)"
            }
        )
        persist_history_to_store()

    except Exception as e:
        st.error(f"Error: {e}")
