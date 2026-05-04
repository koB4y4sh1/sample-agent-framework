from agent_framework.observability import create_resource, enable_instrumentation
from azure.identity import AzureCliCredential
from azure.monitor.opentelemetry import configure_azure_monitor

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

    # Agent Framework 計測
    enable_instrumentation(enable_sensitive_data=False)
