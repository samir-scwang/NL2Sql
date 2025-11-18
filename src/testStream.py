"""
LangChain 1.0 streaming demo for DeepSeek-compatible chat completions.

Prerequisites:
  - pip install langchain langchain-openai python-dotenv
  - Set DEEPSEEK_API_KEY (and optionally DEEPSEEK_API_BASE/DEEPSEEK_MODEL_NAME)
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def _get_env(name: str, default: Optional[str] = None) -> str:
    """Read env vars and raise helpful errors when required ones are missing."""
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Environment variable '{name}' is required but missing.")
    return value


def build_stream_chain():
    """Construct an LCEL chain that yields tokens incrementally."""
    api_key = _get_env("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))

    chat_model = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        timeout=30,
        max_retries=2,
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是专业的 LangChain 导师，用简体中文连续输出答案。"),
            (
                "human",
                "请实时讲解 LangChain 在 {topic} 场景的链式调用要点，并逐步给出示例。",
            ),
        ]
    )

    # Prompt -> model -> text parser; the runnable supports .stream().
    return prompt | chat_model | StrOutputParser()


def stream_topic(topic: str) -> None:
    """Invoke the chain in streaming mode and print chunks as they arrive."""
    chain = build_stream_chain()
    print("开始流式输出：")
    for chunk in chain.stream({"topic": topic}):
        print(chunk, end="", flush=True)
    print("\n--- 流式输出结束 ---")


def main() -> None:
    load_dotenv()
    topic = os.getenv("TEST_TOPIC", "LangChain RAG 项目实践")
    stream_topic(topic)


if __name__ == "__main__":
    main()
