#!/usr/bin/env python3
"""
Test script to verify the hybrid storage fix works correctly.

This tests the core functionality:
1. File storage in natural language format
2. File retrieval using semantic search
3. Task storage in natural language format  
4. Task retrieval using semantic search
"""

import asyncio
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.memory_agent import memory_agent

async def test_hybrid_storage_fix():
    """Test the hybrid storage fix for files and tasks"""
    print("🚀 TESTING HYBRID STORAGE FIX")
    print("=" * 50)
    
    # Test 1: Store a file using the new hybrid format
    print("\n1. Testing File Storage (Hybrid Format)")
    print("-" * 40)
    
    file_id = await memory_agent.save_file_reference_with_context(
        file_path="https://www.figma.com/design/homepage",
        file_type="design",
        purpose="homepage design",
        project="homepage",
        semantic_tags=["Figma", "design", "homepage", "UI"],
        notes="Main homepage design file"
    )
    
    if file_id:
        print(f"✅ File stored successfully with ID: {file_id}")
    else:
        print("❌ File storage failed")
        return False
    
    # Test 2: Retrieve the file using semantic search
    print("\n2. Testing File Retrieval (Semantic Search)")
    print("-" * 40)
    
    # Test different query patterns
    test_queries = [
        "What design files do I have?",
        "homepage file",
        "design files",
        "figma files"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = await memory_agent.find_files_by_intent(query)
        if results:
            print(f"✅ Found {len(results)} files:")
            for file_info in results:
                print(f"   - {file_info['file_path']} ({file_info['file_type']})")
        else:
            print("❌ No files found")
    
    # Test 3: Store a task using the new hybrid format
    print("\n3. Testing Task Storage (Hybrid Format)")
    print("-" * 40)
    
    task_id = await memory_agent.save_task_with_context(
        task="Fix authentication bug",
        priority="urgent",
        due_date="tomorrow",
        project="homepage",
        semantic_tags=["bug", "authentication", "urgent"],
        notes="Critical security issue"
    )
    
    if task_id:
        print(f"✅ Task stored successfully with ID: {task_id}")
    else:
        print("❌ Task storage failed")
        return False
    
    # Test 4: Retrieve tasks using semantic search
    print("\n4. Testing Task Retrieval (Semantic Search)")
    print("-" * 40)
    
    # Test different query patterns
    task_queries = [
        "What tasks do I have?",
        "urgent tasks",
        "homepage tasks",
        "authentication tasks"
    ]
    
    for query in task_queries:
        print(f"\nQuery: '{query}'")
        results = await memory_agent.find_tasks_by_intent(query)
        if results:
            print(f"✅ Found {len(results)} tasks:")
            for task_info in results:
                print(f"   - {task_info['task']} ({task_info['priority']})")
        else:
            print("❌ No tasks found")
    
    print("\n" + "=" * 50)
    print("🎉 HYBRID STORAGE FIX TEST COMPLETED!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    asyncio.run(test_hybrid_storage_fix()) 