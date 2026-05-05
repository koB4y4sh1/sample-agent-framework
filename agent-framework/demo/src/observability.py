from agent_framework.observability import create_resource, enable_instrumentation
from azure.identity import AzureCliCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.mcp import McpInstrumentor

# from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor


def setup_observability() -> None:
    # span_processors: list[SimpleSpanProcessor] = []
    # span_processors.append(SimpleSpanProcessor(ConsoleSpanExporter()))

    # Azure Moniter 構成
    configure_azure_monitor(
        credential=AzureCliCredential(),
        resource=create_resource(service_name="gptapp-demo"),
        enable_live_metrics=True,
        # span_processors=span_processors,
    )

    # OTel の自動計測
    HTTPXClientInstrumentor().instrument()  # httpx
    AioHttpClientInstrumentor().instrument()  # aiohttp
    McpInstrumentor().instrument()  # MCP
    AnthropicInstrumentor().instrument()  # Anthropic
    enable_instrumentation(enable_sensitive_data=False)  # Agent Framework

    # NOTE: OpenAI Instrumentation
    # Agent Framework の Responses API streaming 実行と競合するため使用しない。
    # stream=True の戻り値は AsyncStreamWrapper だが、instrumentation 側が通常レスポンスとして .id を参照し、AttributeError になるケースがある。
    # Agent Framework 側の enable_instrumentation() に寄せて計測する。
