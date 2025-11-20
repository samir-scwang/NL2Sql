import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import redis
from redis import Redis

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "新的会话"
INDEX_KEY = "text2sql:conversation_index"
CONV_KEY_PREFIX = "text2sql:conversation:"


def _conversation_key(conversation_id: str) -> str:
    return f"{CONV_KEY_PREFIX}{conversation_id}"


def _derive_title(messages: List[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            text = str(message.get("content", "")).strip()
            if text:
                return text[:36] + ("…" if len(text) > 36 else "")
    return DEFAULT_TITLE


@dataclass
class ConversationSummary:
    conversation_id: str
    title: str
    updated_at: float


class RedisChatStore:
    """
    负责和 Redis 交互，保存 / 读取会话列表与历史记录。
    设计说明：
      - conversations index：zset（key = INDEX_KEY），score = 更新时间戳
      - conversation data：hash（key = CONV_KEY_PREFIX + conversation_id）
          fields：title, updated_at, messages(JSON 字符串)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        password: Optional[str] = None,
        db: int = 0,
        decode_responses: bool = True,
    ) -> None:
        self.host = host or os.getenv("REDIS_HOST", "192.168.88.130")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.password = password or os.getenv("REDIS_PASSWORD", "redis2025")
        self.db = db
        self.decode_responses = decode_responses
        self._client: Optional[Redis] = None
        self._available = False
        self._lazy_init()

    def _lazy_init(self) -> None:
        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=self.decode_responses,
                socket_timeout=2.0,
            )
            self._client.ping()
            self._available = True
        except Exception as exc:  # pragma: no cover - 网络环境原因
            logger.warning("Redis chat store init failed: %s", exc)
            self._client = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None

    def create_conversation(self, title: Optional[str] = None) -> Optional[str]:
        if not self.is_available:
            return None
        conversation_id = uuid.uuid4().hex
        now = time.time()
        payload_title = title or DEFAULT_TITLE
        try:
            pipe = self._client.pipeline()
            pipe.hset(
                _conversation_key(conversation_id),
                mapping={
                    "title": payload_title,
                    "updated_at": str(now),
                    "messages": "[]",
                },
            )
            pipe.zadd(INDEX_KEY, {conversation_id: now})
            pipe.execute()
            return conversation_id
        except redis.RedisError as exc:  # pragma: no cover - 网络异常
            logger.error("Failed to create conversation: %s", exc)
            return None

    def load_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        if not self.is_available:
            return []
        try:
            data = self._client.hgetall(_conversation_key(conversation_id))
            if not data:
                return []
            raw = data.get("messages") or "[]"
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "Conversation %s has invalid JSON payload, fallback to empty",
                    conversation_id,
                )
                return []
        except redis.RedisError as exc:  # pragma: no cover
            logger.error("Failed to load conversation %s: %s", conversation_id, exc)
            return []

    def list_conversations(self, limit: int = 30) -> List[ConversationSummary]:
        if not self.is_available:
            return []
        try:
            ids = self._client.zrevrange(INDEX_KEY, 0, limit - 1)
            summaries: List[ConversationSummary] = []
            for conv_id in ids:
                meta = self._client.hgetall(_conversation_key(conv_id))
                title = meta.get("title") or DEFAULT_TITLE
                updated_at_str = meta.get("updated_at") or "0"
                try:
                    updated_at = float(updated_at_str)
                except (TypeError, ValueError):
                    updated_at = 0.0
                summaries.append(
                    ConversationSummary(
                        conversation_id=conv_id,
                        title=title,
                        updated_at=updated_at,
                    )
                )
            return summaries
        except redis.RedisError as exc:  # pragma: no cover
            logger.error("Failed to list conversations: %s", exc)
            return []

    def save_messages(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        title: Optional[str] = None,
    ) -> bool:
        if not self.is_available:
            return False
        if not conversation_id:
            return False
        payload = json.dumps(messages, ensure_ascii=False)
        now = time.time()
        payload_title = title or _derive_title(messages)
        try:
            pipe = self._client.pipeline()
            pipe.hset(
                _conversation_key(conversation_id),
                mapping={
                    "messages": payload,
                    "updated_at": str(now),
                    "title": payload_title or DEFAULT_TITLE,
                },
            )
            pipe.zadd(INDEX_KEY, {conversation_id: now})
            pipe.execute()
            return True
        except redis.RedisError as exc:  # pragma: no cover
            logger.error("Failed to save conversation %s: %s", conversation_id, exc)
            return False

    def delete_conversation(self, conversation_id: str) -> bool:
        if not self.is_available:
            return False
        try:
            pipe = self._client.pipeline()
            pipe.delete(_conversation_key(conversation_id))
            pipe.zrem(INDEX_KEY, conversation_id)
            pipe.execute()
            return True
        except redis.RedisError as exc:  # pragma: no cover
            logger.error("Failed to delete conversation %s: %s", conversation_id, exc)
            return False
