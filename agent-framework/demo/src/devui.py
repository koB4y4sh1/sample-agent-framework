from __future__ import annotations

from agent_framework.devui import serve
from app import DemoApplication, DemoConfig
from settings import load_model_settings_list


def select_model_by_input() -> str:
    models = load_model_settings_list()
    if not models:
        raise ValueError("No model settings were found in settings/model.json.")

    print("Available models:")
    for index, model in enumerate(models, start=1):
        print(f"{index}. {model.provider_family}: {model.model_name}")

    while True:
        value = input("Select model number: ").strip()
        if value.isdigit():
            index = int(value)
            if 1 <= index <= len(models):
                return models[index - 1].model_name
        print("Invalid selection.")

def main() -> None:

    model = select_model_by_input()
    app = DemoApplication(
        config=DemoConfig(model=model),
    )

    serve(
        entities=[app.agent],
        host="127.0.0.1",
        port=8080,
        auto_open=True,
        ui_enabled=True,
        instrumentation_enabled=True, # OTelを有効化
        mode="developer",
    )


if __name__ == "__main__":
    main()
