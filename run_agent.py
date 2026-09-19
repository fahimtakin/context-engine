# run_agent.py
import asyncio
from langchain_core.messages import HumanMessage
from app.agents.graph import agent_brain, redis_saver_context

async def run_conversation():
    async with redis_saver_context as saver:
        
        # Configure our Session/Thread ID tracking block
        config = {
            "configurable": {
                "thread_id": "production-simulation-2026"
            },
            "checkpointer": saver
        }

        # --- Turn 1: Introduction ---
        print("\n=== Turn 1: Introducing Ourselves ===")
        # On the very first turn, we pass the initial state setup dictionary
        input_1 = {"messages": [HumanMessage(content="Hello! My name is Alex and I am a software engineer interviewing for an AI role.")]}
        
        output_1 = await agent_brain.ainvoke(input_1, config=config)
        print(f"Agent Response:\n{output_1.get('final_output')}\n")

        # --- Turn 2: The Follow-Up Question ---
        print("=== Turn 2: Testing Multi-Turn Context ===")
        # Instead of passing a manual, empty history block, 
        # we pass *only* the new message inside the update payload dictionary. 
        # LangGraph will pull the historic 'messages' list automatically out of Redis!
        input_2 = {"messages": [HumanMessage(content="What was my name again, and what role am I pursuing?")]}
        
        output_2 = await agent_brain.ainvoke(input_2, config=config)
        print(f"Agent Response:\n{output_2.get('final_output')}\n")

if __name__ == "__main__":
    asyncio.run(run_conversation())
