import uuid
import json
import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# =========================
# 0. 环境变量 & 模型初始化
# =========================

load_dotenv()


def _get_env(name: str, default: Optional[str] = None) -> str:
    """Return environment variable values with a clearer error message."""
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Environment variable '{name}' is required but missing.")
    return value


api_key = _get_env("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))

deepseek = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model=model_name,
    temperature=temperature,
    timeout=120,
    max_retries=2,
)


# =========================
# 1. 定义一个假的 send_email_tool
# =========================
@tool
def send_email_tool(to: str, subject: str, body: str) -> str:
    """真正发送邮件的工具。
    当用户要求‘发邮件’、‘发信给某人’时，在获得足够信息后调用这个工具，
    而不是只把邮件内容返回给用户。
    参数:
      - to: 收件人邮箱
      - subject: 邮件主题
      - body: 邮件正文
    """
    # 真实场景下这里可以集成 SMTP / 邮件服务商
    print(f"[send_email_tool] Sending email to={to}, subject={subject}")
    print(body)
    return f"Email sent to {to} with subject '{subject}'."


# =========================
# 2. 初始化 Streamlit 状态（含 checkpointer）
# =========================

if "checkpointer" not in st.session_state:
    # 🔴 非常关键：checkpointer 要在整个会话中保持同一个实例
    st.session_state["checkpointer"] = InMemorySaver()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    # 只用于前端展示（Agent 自己用 checkpointer 记忆）
    st.session_state["messages"] = []

if "pending_interrupt" not in st.session_state:
    # 保存 HITL 请求（action_requests 等）
    st.session_state["pending_interrupt"] = None


# =========================
# 3. 创建 Agent + Human-in-the-loop 中间件
# =========================
def get_agent():
    checkpointer = st.session_state["checkpointer"]

    hitl = HumanInTheLoopMiddleware(
        interrupt_on={
            # 对 send_email_tool 进行人工审批
            "send_email_tool": {
                "allowed_decisions": ["approve", "reject"],
            }
        },
        description_prefix="需要人工审批的工具调用：",
    )

    # 可以按需加上 state_modifier，让模型更愿意调用工具
    agent = create_agent(
        model=deepseek,
        tools=[send_email_tool],
        middleware=[hitl],
        checkpointer=checkpointer,
    )
    return agent


agent = get_agent()

# =========================
# 4. Streamlit UI
# =========================

st.title("📧 Human-in-the-loop 邮件 Agent Demo")

st.write(
    "示例：让 Agent 起草邮件、调用 `send_email_tool` 之前，"
    "必须通过 Human-in-the-loop 中间件进行人工审批。"
)

# ---- 显示历史对话 ----
st.subheader("对话历史")

for role, content in st.session_state["messages"]:
    if role == "user":
        st.markdown(f"**🧑 用户：** {content}")
    elif role == "assistant":
        st.markdown(f"**🤖 Agent：** {content}")
    elif role == "system":
        st.markdown(f"**⚙️ 系统：** {content}")

st.markdown("---")

config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

# =========================
# 5. 用户输入（仅在没有 pending interrupt 时可用）
# =========================

if st.session_state["pending_interrupt"] is None:
    user_input = st.text_input(
        "输入你的指令（例如：帮我给老板写一封道歉邮件）",
        key="user_input",
    )

    if st.button("发送", disabled=not user_input.strip()):
        st.session_state["messages"].append(("user", user_input))

        # 调用 agent，一直跑到完成或遇到 interrupt
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ]
            },
            config=config,
        )

        # 情况 1：被 Human-in-the-loop 中断
        if "__interrupt__" in result:
            interrupt_list = result["__interrupt__"]
            interrupt = interrupt_list[0]  # 这里假设只有一个

            hitl_value = interrupt.value  # dict: {action_requests, review_configs}
            st.session_state["pending_interrupt"] = hitl_value

            st.session_state["messages"].append(
                (
                    "system",
                    "⚠️ Agent 触发了需要人工审批的工具调用，等待你的决定。",
                )
            )

        # 情况 2：正常完成，没有中断
        else:
            if "messages" in result and len(result["messages"]) > 0:
                last_msg = result["messages"][-1]
                # last_msg 可能是 dict / BaseMessage，这里做个兼容
                if isinstance(last_msg, dict):
                    content = last_msg.get("content", "")
                else:
                    content = getattr(last_msg, "content", str(last_msg))

                st.session_state["messages"].append(("assistant", content))

        st.rerun()

else:
    st.warning("当前有一个待审批的工具调用，请先处理。")

# =========================
# 6. 如果有 pending interrupt，展示审批 UI
# =========================

if st.session_state["pending_interrupt"] is not None:
    st.markdown("---")
    st.subheader("🛑 人工审批区")

    interrupt_value = st.session_state["pending_interrupt"]
    action_requests = interrupt_value["action_requests"]
    review_configs = interrupt_value["review_configs"]

    # 为简单起见，我们假设一次只拦一个工具调用
    action = action_requests[0]
    review_config = review_configs[0]

    tool_name = action["name"]
    # HumanInTheLoopMiddleware 暴露的是 "arguments"
    args = action.get("arguments", {})  # 防御性写法
    allowed = review_config["allowed_decisions"]

    st.markdown(f"**待审批工具**：`{tool_name}`")
    st.markdown("**工具参数：**")
    st.code(json.dumps(args, indent=2, ensure_ascii=False))

    st.markdown(f"**允许的决策类型**：`{allowed}`")

    decision = st.radio(
        "请选择你的决策：",
        options=allowed,
        key="decision_radio",
    )

    reject_reason = ""
    if decision == "reject":
        reject_reason = st.text_area(
            "如选择拒绝，可以填写拒绝原因（可选）",
            key="reject_reason",
        )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("确认提交决策"):
            # 拼装 decisions 列表（这里只有一个）
            decision_obj = {"type": decision}
            if decision == "reject" and reject_reason.strip():
                decision_obj["feedback"] = reject_reason.strip()

            cmd = Command(
                resume={
                    "decisions": [decision_obj],
                }
            )

            # 用同一个 thread_id 恢复执行
            result = agent.invoke(cmd, config=config)

            # 清空 pending_interrupt
            st.session_state["pending_interrupt"] = None

            # 追加系统提示 + 最终回复
            st.session_state["messages"].append(
                ("system", f"你对 `{tool_name}` 的决策是：{decision_obj}")
            )

            if "messages" in result and len(result["messages"]) > 0:
                last_msg = result["messages"][-1]
                if isinstance(last_msg, dict):
                    content = last_msg.get("content", "")
                else:
                    content = getattr(last_msg, "content", str(last_msg))
                st.session_state["messages"].append(("assistant", content))

            st.rerun()

    with col2:
        if st.button("取消（不做任何操作）"):
            st.session_state["pending_interrupt"] = None
            st.rerun()
