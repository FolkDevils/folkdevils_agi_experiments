#!/usr/bin/env python3
"""
Definitive test to prove the catastrophic feedback loop is permanently fixed.
"""

import asyncio
import sys
import os

# Ensure the project root is in the Python path
sys.path.append('/Users/andreweaton/son-of-andrew')

from agents.meta_agent import meta_agent
from agents.memory_agent import memory_agent
from purge_memory import purge_memory

# Clean up previous test runs
async def cleanup_memory():
    """Remove any test-related facts from memory before running."""
    print("\n--- 🧹 CLEANING MEMORY ---")
    # Clean up file references
    await memory_agent.forget_fact_aggressively("figma.com/design/homepage-redesign-definitive")
    # Clean up false negative facts
    await memory_agent.forget_fact_aggressively("has not saved any references for homepage")
    print("--- ✅ MEMORY CLEANED ---\n")

async def test_definitive_fix():
    """
    This test verifies the complete, two-part architectural fix:
    1.  Search is hardened and finds the file correctly.
    2.  Opportunistic learning does NOT store a false negative fact.
    """
    print("🧪 Firing up the definitive test for the feedback loop fix...")
    print("============================================================")

    # STEP 1: Store a file reference to ensure data exists.
    # This simulates the user mentioning a file they are working on.
    print("\n1️⃣ STEP 1: Storing a file reference...")
    store_input = "I am working on the final design here: https://www.figma.com/design/homepage-redesign-definitive"
    store_state = {
        "user_input": store_input,
        "session_id": "definitive_test_session"
    }
    store_result = await meta_agent.process_request(**store_state)
    print(f"   - Store Response: {store_result.get('content')}")
    assert "I'll remember that you're working with" in store_result.get('content', '')
    print("   - ✅ File reference stored successfully.")

    # Give a moment for async memory operations to complete
    await asyncio.sleep(2)

    # STEP 2: Query for the file that was just stored.
    # This is where the original failure occurred.
    print("\n2️⃣ STEP 2: Querying for the stored file...")
    query_input = "What files am I working on for homepage?"
    query_state = {
        "user_input": query_input,
        "session_id": "definitive_test_session"
    }
    query_result = await meta_agent.process_request(**query_state)
    query_response = query_result.get('content', '')
    print(f"   - Query Response: {query_response}")
    
    # VERIFY that the file was found
    assert "https://www.figma.com/design/homepage-redesign-definitive" in query_response
    print("   - ✅ File was successfully found by the search.")

    # Give a moment for the async opportunistic learning to (not) run
    await asyncio.sleep(2)

    # STEP 3: Verify that the destructive "false negative" fact was NOT created.
    # This is the most critical part of the test.
    print("\n3️⃣ STEP 3: Verifying that NO destructive learning occurred...")
    facts = await memory_agent.search_facts("has not saved any references for homepage")
    
    if not facts:
        print("   - ✅ SUCCESS: The destructive 'no files found' fact was NOT created.")
    else:
        print("   - ❌ FAILURE: A destructive 'no files found' fact was created:")
        for fact in facts:
            print(f"     - {fact.get('content')}")
        assert not facts, "Destructive learning occurred, poisoning the memory."

    print("\n============================================================")
    print("🎉 CONGRATULATIONS: The catastrophic feedback loop is officially fixed.")
    print("============================================================\n")


if __name__ == "__main__":
    async def main():
        # Purge memory first
        await purge_memory()
        # Now run the test
        await test_definitive_fix()
    
    asyncio.run(main()) 