from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid8

from agent_framework import Message
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential

from .base import MessageStore


class CosmosStore(MessageStore):
    """Azure Cosmos DB に会話履歴を保存するストア。"""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        key: str | None = None,
        database_id: str = "agent_data",
        container_id: str = "history_messages",
        tenant_id: str = "default",
    ) -> None:
        endpoint_value = endpoint or os.getenv("COSMOS_ENDPOINT", "").strip()
        key_value = key or os.getenv("COSMOS_KEY", "").strip()
        if not endpoint_value:
            raise ValueError(
                "CosmosStore の初期化に失敗しました。COSMOS_ENDPOINT を設定してください。"
            )

        self._tenant_id = tenant_id.strip() or "default"
        credential = key_value if key_value else AzureCliCredential()
        self._client = CosmosClient(endpoint_value, credential=credential)
        database = self._client.get_database_client(database_id)
        self._container = database.get_container_client(container_id)

    async def read_messages(self, session_id: str | None) -> list[Message]:
        safe_session_id = self._safe_session_id(session_id)
        pk = self._partition_key(safe_session_id)
        pager = self._container.query_items(
            query=(
                "SELECT c.message FROM c "
                "WHERE c.tenantId = @tenantId AND c.sessionId = @sessionId "
                "ORDER BY c.sequence ASC"
            ),
            parameters=[
                {"name": "@tenantId", "value": self._tenant_id},
                {"name": "@sessionId", "value": safe_session_id},
            ],
            partition_key=pk,
        )
        items = [item async for item in pager]
        return [Message.from_dict(item["message"]) for item in items]

    async def write_messages(
        self, session_id: str | None, messages: Sequence[Message]
    ) -> None:
        safe_session_id = self._safe_session_id(session_id)
        pk = self._partition_key(safe_session_id)
        existing_pager = self._container.query_items(
            query="SELECT c.id FROM c WHERE c.tenantId = @tenantId AND c.sessionId = @sessionId",
            parameters=[
                {"name": "@tenantId", "value": self._tenant_id},
                {"name": "@sessionId", "value": safe_session_id},
            ],
            partition_key=pk,
        )
        existing_items = [item async for item in existing_pager]
        for item in existing_items:
            await self._container.delete_item(item=item["id"], partition_key=pk)

        now = datetime.now(UTC).isoformat()
        for sequence, message in enumerate(messages):
            await self._container.upsert_item(
                {
                    "id": f"msg_{uuid8().hex}",
                    "pk": pk,
                    "tenantId": self._tenant_id,
                    "sessionId": safe_session_id,
                    "sequence": sequence,
                    "createdAt": now,
                    "message": message.to_dict(),
                }
            )

    async def close(self) -> None:
        await self._client.close()

    def _safe_session_id(self, session_id: str | None) -> str:
        return (session_id or "default").replace("/", "_").replace("\\", "_")

    def _partition_key(self, safe_session_id: str) -> str:
        return f"{self._tenant_id}|{safe_session_id}"
