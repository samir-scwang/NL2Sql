import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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
