#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.precision_editor_agent import PrecisionEditorAgent
from agents.working_text_manager import working_text_manager

async def debug_semantic_matching():
    """Debug semantic matching issue"""
    
    test_session = "debug_semantic"
    agent = PrecisionEditorAgent()
    
    # Clear any existing data
    await working_text_manager.clear_working_text(test_session)
    
    # Set up test content  
    test_content = "I don't wanna have bad energy when I'm talking with Avi."
    
    print(f"📝 Setting working text: {test_content}")
    
    # Step 1: Set working text
    result = await agent.ainvoke({
        "session_id": test_session,
        "new_content": test_content
    })
    
    print(f"✅ Set working text result: {result.name}")
    print(f"   Content: {result.get_user_content()}")
    
    # Step 2: Try to remove with exact match first
    print(f"\n🔍 Trying exact match: remove 'when I'm talking with Avi'")
    result = await agent.ainvoke({
        "session_id": test_session,
        "user_instruction": "remove 'when I'm talking with Avi'"
    })
    
    print(f"✅ Exact match result: {result.name}")
    print(f"   Content: {result.get_user_content()}")
    print(f"   Reason: {result.reason}")
    
    # Step 3: Try to remove with variation (should trigger semantic matching)
    print(f"\n🔍 Trying semantic match: remove 'when talking with Avi'")
    result = await agent.ainvoke({
        "session_id": test_session,
        "user_instruction": "remove 'when talking with Avi'"
    })
    
    print(f"✅ Semantic match result: {result.name}")
    print(f"   Content: {result.get_user_content()}")
    print(f"   Reason: {result.reason}")
    print(f"   State: {result.state}")

if __name__ == "__main__":
    asyncio.run(debug_semantic_matching()) 