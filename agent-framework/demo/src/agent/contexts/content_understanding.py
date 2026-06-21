from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from agent_framework import Content
from agent_framework_azure_contentunderstanding import (
    AnalysisSection,
    ContentUnderstandingContextProvider as AzureContentUnderstandingContextProvider,
    FileSearchConfig,
)
from azure.ai.contentunderstanding.aio import ContentUnderstandingClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureCliCredential as SyncAzureCliCredential
from azure.identity.aio import AzureCliCredential
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, ContentSettings
from azure.storage.blob import generate_blob_sas

DEMO_ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = DEMO_ROOT_DIR / ".input"


@dataclass(frozen=True, slots=True)
class ContentUnderstandingInputConfig:
    """CU に渡す添付ファイルの渡し方を環境変数から決める設定。"""

    use_input_file_data: bool = True
    save_local: bool = True
    local_input_dir: Path = DEFAULT_INPUT_DIR
    blob_container: str = "cu-input"
    blob_prefix: str = "uploads"
    blob_sas_minutes: int = 60

    @classmethod
    def from_env(cls) -> "ContentUnderstandingInputConfig":
        """未設定なら従来どおり `Content.from_data(...)` を使う。"""
        return cls(
            use_input_file_data=_bool_from_env(
                "CU_USE_INPUT_FILE_DATA",
                fallback_name="USE_INPUT_FILE_DATA",
                default=True,
            ),
            save_local=_bool_from_env(
                "CU_SAVE_LOCAL",
                fallback_name="SAVE_LOCAL",
                default=True,
            ),
            local_input_dir=Path(
                os.getenv("CU_LOCAL_INPUT_DIR")
                or os.getenv("LOCAL_INPUT_DIR")
                or str(DEFAULT_INPUT_DIR)
            ),
            blob_container=os.getenv("CU_STORAGE_CONTAINER") or "cu-input",
            blob_prefix=os.getenv("CU_STORAGE_BLOB_PREFIX") or "uploads",
            blob_sas_minutes=int(os.getenv("CU_STORAGE_BLOB_SAS_MINUTES") or "60"),
        )


class ContentUnderstandingContextProvider(AzureContentUnderstandingContextProvider):
    """CU の分析結果を LLM の入力コンテキストに追加する Provider。"""

    DEFAULT_SOURCE_ID = "content_understanding"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        analyzer_id: str | None = None,
        client: ContentUnderstandingClient | None = None,
        credential=None,
        api_key: str | None = None,
        max_wait: float | None = None,
        output_sections: list[AnalysisSection] | None = None,
        file_search: FileSearchConfig | None = None,
        source_id: str | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        resolved_credential = credential
        if client is None and resolved_credential is None:
            # API Key があればそれを優先し、なければ Azure CLI ログインを使う。
            resolved_credential = self._credential_from_api_key(
                api_key
            ) or AzureCliCredential(process_timeout=30)
        super().__init__(
            endpoint=endpoint,
            client=client,
            credential=resolved_credential,
            analyzer_id=analyzer_id,
            max_wait=max_wait,
            output_sections=output_sections,
            file_search=file_search,
            source_id=source_id or self.DEFAULT_SOURCE_ID,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )

    def _credential_from_api_key(self, api_key: str | None):
        resolved_api_key = api_key or os.getenv("AZURE_CONTENTUNDERSTANDING_API_KEY")
        if not resolved_api_key:
            return None
        return AzureKeyCredential(resolved_api_key)


def create_content_understanding_context_provider_from_env(
    *,
    source_id: str | None = None,
    file_search: FileSearchConfig | None = None,
) -> ContentUnderstandingContextProvider | None:
    """CU の endpoint がある場合だけ Provider を作る。"""
    endpoint = os.getenv("AZURE_CONTENTUNDERSTANDING_ENDPOINT")
    if not endpoint:
        return None

    return ContentUnderstandingContextProvider(
        endpoint=endpoint,
        analyzer_id=os.getenv("AZURE_CONTENTUNDERSTANDING_ANALYZER_ID") or None,
        max_wait=_float_from_env("AZURE_CONTENTUNDERSTANDING_MAX_WAIT", default=5.0),
        file_search=file_search,
        source_id=source_id,
    )


def create_cu_attachment_content(
    *,
    name: str,
    media_type: str,
    data: bytes,
    config: ContentUnderstandingInputConfig | None = None,
) -> Content:
    """添付ファイルを data / file URI / Blob URL のいずれかに変換する。"""
    resolved_config = config or ContentUnderstandingInputConfig.from_env()
    additional_properties = {"filename": name}

    if resolved_config.use_input_file_data:
        return Content.from_data(
            data=data,
            media_type=media_type,
            additional_properties=additional_properties,
        )

    if resolved_config.save_local:
        return Content.from_uri(
            _save_local_file(
                name=name,
                data=data,
                input_dir=resolved_config.local_input_dir,
            ),
            media_type=media_type,
            additional_properties=additional_properties,
        )

    return Content.from_uri(
        _save_blob_file(
            name=name,
            media_type=media_type,
            data=data,
            config=resolved_config,
        ),
        media_type=media_type,
        additional_properties=additional_properties,
    )


def _save_local_file(*, name: str, data: bytes, input_dir: Path) -> str:
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / _safe_storage_name(name)
    path.write_bytes(data)
    return path.resolve().as_uri()


def _save_blob_file(
    *,
    name: str,
    media_type: str,
    data: bytes,
    config: ContentUnderstandingInputConfig,
) -> str:
    connection_string = os.getenv("CU_STORAGE_CONNECTION_STRING") or os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING"
    )
    account_url = os.getenv("CU_STORAGE_ACCOUNT_URL")

    if connection_string:
        service_client = BlobServiceClient.from_connection_string(connection_string)
    elif account_url:
        # Blob SDK は同期クライアントなので、ここだけ同期版 AzureCliCredential を使う。
        service_client = BlobServiceClient(
            account_url=account_url,
            credential=SyncAzureCliCredential(process_timeout=30),
        )
    else:
        raise ValueError(
            "Blob upload requires CU_STORAGE_CONNECTION_STRING, "
            "AZURE_STORAGE_CONNECTION_STRING, or CU_STORAGE_ACCOUNT_URL."
        )

    container_client = service_client.get_container_client(config.blob_container)
    if _bool_from_env("CU_STORAGE_CREATE_CONTAINER", default=True):
        try:
            container_client.create_container()
        except Exception:
            # 既に存在する場合も例外になるため、アップロードで最終確認する。
            pass

    blob_name = f"{config.blob_prefix.rstrip('/')}/{_safe_storage_name(name)}"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        data,
        overwrite=True,
        content_settings=ContentSettings(content_type=media_type),
    )

    return _blob_sas_url(
        blob_client=blob_client,
        container_name=config.blob_container,
        blob_name=blob_name,
        minutes=config.blob_sas_minutes,
    ) or blob_client.url


def _blob_sas_url(
    *,
    blob_client,
    container_name: str,
    blob_name: str,
    minutes: int,
) -> str | None:
    account_key = getattr(blob_client.credential, "account_key", None)
    account_name = getattr(blob_client, "account_name", None)
    if not account_key or not account_name:
        return None

    sas = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(minutes=minutes),
    )
    return f"{blob_client.url}?{sas}"


def _safe_storage_name(name: str) -> str:
    safe_name = Path(name).name.replace("\\", "_").replace("/", "_")
    return f"{uuid4().hex}_{safe_name or 'attachment'}"


def _bool_from_env(
    name: str,
    *,
    fallback_name: str | None = None,
    default: bool,
) -> bool:
    value = os.getenv(name)
    if value is None and fallback_name is not None:
        value = os.getenv(fallback_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_from_env(name: str, *, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


__all__ = [
    "ContentUnderstandingContextProvider",
    "ContentUnderstandingInputConfig",
    "create_content_understanding_context_provider_from_env",
    "create_cu_attachment_content",
]
