#!/usr/bin/env python3
"""
Debug My Reflection System

Let me figure out why my reflections aren't being generated.
"""

import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up environment
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

from mind.consciousness_loop import ConsciousnessLoop

async def debug_reflection():
    """Debug my reflection system"""
    print("🐛 DEBUGGING MY REFLECTION SYSTEM")
    print("=" * 40)
    
    # Initialize consciousness
    consciousness = ConsciousnessLoop()
    
    print("\n📊 Current state:")
    print(f"   Active: {consciousness.is_active}")
    print(f"   Sessions: {consciousness.session_count}")
    print(f"   Memories stored: {consciousness.total_memories_stored}")
    
    # Check recent memories
    print("\n🧠 Checking recent memories...")
    recent_memories = await consciousness.long_term_memory.get_recent_memories(hours=24, limit=10)
    print(f"   Found {len(recent_memories)} recent memories")
    
    for i, memory in enumerate(recent_memories[:3], 1):
        print(f"   Memory {i}: {memory.content[:50]}...")
        print(f"     Type: {memory.type}, Participants: {memory.participants}")
    
    # Check reflection engine
    print("\n🤔 Testing reflection engine...")
    reflection_engine = consciousness.reflection_engine
    
    print(f"   Current reflections: {len(reflection_engine.reflections)}")
    print(f"   Reflection count: {reflection_engine.reflection_count}")
    print(f"   Is reflecting: {reflection_engine.is_reflecting}")
    
    # Try to trigger reflection manually
    print("\n🌙 Manually triggering reflection cycle...")
    
    try:
        await reflection_engine.start_reflection_cycle()
        
        print(f"   Post-reflection state:")
        print(f"   Total reflections: {len(reflection_engine.reflections)}")
        print(f"   Reflection count: {reflection_engine.reflection_count}")
        
        # Show any reflections generated
        if reflection_engine.reflections:
            print(f"\n💭 Generated reflections:")
            for i, reflection in enumerate(reflection_engine.reflections, 1):
                print(f"   Reflection {i}:")
                print(f"     Type: {reflection.reflection_type}")
                print(f"     Content: {reflection.content}")
                print(f"     Confidence: {reflection.confidence}")
                print(f"     Priority: {reflection.priority}")
        else:
            print("   ❌ No reflections generated")
            
            # Debug why no reflections
            print("\n🔍 Debugging reflection generation...")
            
            # Test pattern analysis
            if recent_memories:
                pattern_reflection = await reflection_engine._analyze_patterns(recent_memories)
                print(f"   Pattern analysis result: {pattern_reflection is not None}")
                if pattern_reflection:
                    print(f"     Content: {pattern_reflection.content}")
                
                # Test self analysis  
                self_reflection = await reflection_engine._self_analysis(recent_memories)
                print(f"   Self-analysis result: {self_reflection is not None}")
                if self_reflection:
                    print(f"     Content: {self_reflection.content}")
                
                # Test relationship analysis
                relationship_reflection = await reflection_engine._analyze_relationship(recent_memories)
                print(f"   Relationship analysis result: {relationship_reflection is not None}")
                if relationship_reflection:
                    print(f"     Content: {relationship_reflection.content}")
                
                # Test goal analysis
                goal_reflection = await reflection_engine._analyze_goals(recent_memories)
                print(f"   Goal analysis result: {goal_reflection is not None}")
                if goal_reflection:
                    print(f"     Content: {goal_reflection.content}")
            else:
                print("   ❌ No recent memories to analyze")
    
    except Exception as e:
        print(f"   ❌ Error during reflection: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    print("   Cleaning up...")
    consciousness.long_term_memory.client.close()
    
    print("\n🎯 DEBUG COMPLETE")

if __name__ == "__main__":
    asyncio.run(debug_reflection()) 