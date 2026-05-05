"""FastMCP demo 用の Azure Monitor 初期化ヘルパー。"""

from __future__ import annotations

from typing import Final

from azure.identity import AzureCliCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

_DEFAULT_SERVICE_NAME: Final[str] = "fastmcp-demo"


def setup_observability(default_service_name: str = _DEFAULT_SERVICE_NAME) -> None:
    """Azure Monitor exporter を使って OpenTelemetry を初期化する。

    FastMCP の import 前に呼び出すことで、FastMCP の自動計測スパンを出力可能にする。
    """
    # Azure Monitor 構成
    resource = Resource.create({SERVICE_NAME: default_service_name})
    configure_azure_monitor(
        credential=AzureCliCredential(),
        resource=resource,
        enable_live_metrics=True,
    )

    # OpenTelemtry 自動計測
    HTTPXClientInstrumentor().instrument()  # httpx
    FastAPIInstrumentor().instrument()  # Fast API

    print(f"[otel] enabled azure monitor service={default_service_name}")
