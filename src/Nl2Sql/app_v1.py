# src/Nl2Sql/app.py
# -*- coding: utf-8 -*-
# streamlit run D:\pythonProject\Text2Sql\src\Nl2Sql\app.py
import sys
from pathlib import Path
import json
from typing import Dict, Any

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
    log_expander = st.expander("🛠️ 工具执行过程（ToolRuntime Stream）", expanded=True)
    updates_expander = st.expander("🧩 Agent 内部状态更新（调试用）", expanded=False)

    log_placeholder = log_expander.empty()
    updates_placeholder = updates_expander.empty()

    # 4. 缓冲区
    log_buffer = ""
    updates_buffer = ""
    final_answer = ""   # 用于记录“最新的 AI 回复”

    try:
        # 使用和你 __main__ 里一样的 stream 调用方式
        for stream_mode, chunk in agent.stream(
            {
                "messages": [{"role": "user", "content": user_input}],
                "user_name": "web_user",
            },
            stream_mode=["updates", "custom"],
        ):
            # ---------- custom：ToolRuntime.stream_writer 输出 ----------
            if stream_mode == "custom":
                text = str(chunk)
                log_buffer += text + "\n"
                log_placeholder.code(log_buffer, language="text")

            # ---------- updates：LangGraph 状态更新 ----------
            elif stream_mode == "updates":
                # 1）调试信息展示
                try:
                    pretty = json.dumps(chunk, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    pretty = str(chunk)

                updates_buffer += pretty + "\n" + "-" * 60 + "\n"
                updates_placeholder.code(updates_buffer, language="json")

                # 2）尝试从 updates 中提取 AIMessage，实时展示模型当前回复
                if isinstance(chunk, dict) and "model" in chunk:
                    model_part: Dict[str, Any] = chunk["model"]  # type: ignore
                    msgs = model_part.get("messages") or []
                    if isinstance(msgs, list) and msgs:
                        last_msg = msgs[-1]
                        if isinstance(last_msg, AIMessage):
                            if isinstance(last_msg.content, str):
                                final_answer = last_msg.content
                                # 更新对话里的“助手回答”区域
                                answer_placeholder.markdown(final_answer)

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
