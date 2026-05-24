import argparse
import asyncio
import json

import httpx
from a2a.client import A2ACardResolver
from a2a.server.request_handlers.response_helpers import agent_card_to_dict


async def main() -> None:
    """Agent Card を取得し、公開されている capability と skill を表示します。"""
    parser = argparse.ArgumentParser(description="Agent Card capability client")
    parser.add_argument("--url", default="http://127.0.0.1:41242")
    args = parser.parse_args()

    async with httpx.AsyncClient() as httpx_client:
        # Agent Card は /.well-known/agent-card.json から取得されます。
        card = await A2ACardResolver(httpx_client, args.url).get_agent_card()

    print(f"name: {card.name}")
    print(f"streaming: {card.capabilities.streaming}")
    print(f"push_notifications: {card.capabilities.push_notifications}")
    print("skills:")
    for skill in card.skills:
        print(f"- {skill.id}: {skill.description}")
    print("Agent Card Json:")
    card_dict = agent_card_to_dict(card)
    print(json.dumps(card_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
