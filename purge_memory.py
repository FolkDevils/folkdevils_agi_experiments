#!/usr/bin/env python3
"""
Purge script to aggressively wipe specific toxic facts from memory.
"""

import asyncio
import sys

sys.path.append('/Users/andreweaton/son-of-andrew')

from memory_manager import memory_manager

async def purge_memory():
    """Aggressively purge all facts related to the test."""
    print("--- 🗑️ PURGING TAINTED MEMORY ---")
    
    # This requires direct interaction with the Zep client, which is abstracted.
    # For this test, we will use the existing forget_fact and hope it clears the index.
    # A more robust solution would be a direct Zep call to delete from a session.
    
    # Purge file references
    await memory_manager.forget_fact_aggressively("figma.com/design/homepage-redesign-definitive")
    
    # Purge the false negative facts
    await memory_manager.forget_fact_aggressively("has not saved any references for homepage")
    
    # Purge any deletion records of the false negative facts
    await memory_manager.forget_fact_aggressively("DELETED_FACT: Andrew has not saved any references")

    print("--- ✅ MEMORY PURGED ---")

if __name__ == "__main__":
    asyncio.run(purge_memory()) 