from __future__ import annotations

from random import randint
from typing import Annotated

from agent_framework import tool


@tool(approval_mode="always_require")
def get_weather(
    location: Annotated[str, "Weather target city such as Tokyo, New York, Paris"],
) -> str:
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"{location} weather is {conditions[randint(0, 3)]}, approx. {randint(10, 30)}C."
