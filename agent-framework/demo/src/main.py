import asyncio
 
from app import DemoApplication, DemoConfig
from chat_cli import DemoChatCLI
 
 
async def main() -> None:
    # モデル選択
    selected_model = DemoChatCLI.select_model()
 
    # アプリケーション構築
    app = DemoApplication(config=DemoConfig(model=selected_model))
 
    # セッション開始
    session = app.create_session()
 
    # CLI 起動
    cli = DemoChatCLI(
        agent=app.agent,
        session=session,
        code_interpreter_status=app.skills.describe(),
        stream_renderer=app.stream_renderer,
    )
    await cli.run()
 
 
if __name__ == "__main__":
    asyncio.run(main())