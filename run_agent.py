"""Two-turn demo of the agent from the command line, showing memory across turns."""

import asyncio

from langchain_core.messages import HumanMessage

from app.agents.graph import compile_agent, open_checkpointer
from app.core.config import settings
from app.core.logging_setup import configure_logging

TURNS = [
    "Hello! My name is Alex and I am a software engineer interviewing for an AI role.",
    "What was my name again, and what role am I pursuing?",
]


async def main() -> None:
    configure_logging(settings.LOG_LEVEL)
    config = {"configurable": {"thread_id": "cli-demo"}}
    async with open_checkpointer() as saver:
        agent = compile_agent(saver)
        for i, text in enumerate(TURNS, start=1):
            state = await agent.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)
            print(f"\n=== Turn {i} [{state.get('router_decision')}] ===")
            print(f"You:   {text}")
            print(f"Agent: {state.get('final_output')}")


if __name__ == "__main__":
    asyncio.run(main())
