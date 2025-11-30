import asyncio
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.observability import setup_observability,get_tracer, get_meter
from azure.identity import AzureCliCredential
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

# opentelemetry-api
# opentelemetry-sdk
# opentelemetry-exporter-otlp-proto-grpc
# opentelemetry-semantic-conventions-ai

# Azure Monitor (Application Insights) にエクスポートする場合
# pip install azure-monitor-opentelemetry

# 1. 初期設定
# 1-1. 環境変数の読み取り
setup_observability()

# 1-2. 環境変数ではなく、プログラムで構成する場合
setup_observability(
    enable_sensitive_data=True,
    otlp_endpoint="http://localhost:4317",
    applicationinsights_connection_string="InstrumentationKey=your_key"
)

# 1-3. それ以外の高度な設定方法
custom_exporters = [
    OTLPSpanExporter(endpoint="http://localhost:4317"),
    ConsoleSpanExporter()
]
setup_observability(exporters=custom_exporters, enable_sensitive_data=True)

# 2. カスタム スパンとメトリックを作成する場合
tracer = get_tracer()
meter = get_meter()
with tracer.start_as_current_span("my_custom_span"):
    # Your code here
    pass
counter = meter.create_counter("my_custom_counter")
counter.add(1, {"key": "value"})

async def main():
    # 3. エージェントの作成
    agent = ChatAgent(
        chat_client=AzureOpenAIChatClient(
            credential=AzureCliCredential(),
        ),
        name="Joker",
        instructions="You are good at telling jokes."
    )

    # 4 エージェントの実行
    result = await agent.run("Tell me a joke about a pirate.")
    print(result.text)


asyncio.run(main())
