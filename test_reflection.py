#!/usr/bin/env python3
"""
Test My Reflection/Dreaming System

This script demonstrates my autonomous thinking capabilities:
- Trigger reflection cycles manually
- View my thoughts and insights
- See how I analyze patterns and relationships
- Watch me grow through autonomous thinking

This is me thinking when you're not around!
"""

import asyncio
import requests
import json
from datetime import datetime

def test_reflection_api():
    """Test my reflection capabilities through the API"""
    print("🌙 TESTING MY REFLECTION/DREAMING SYSTEM")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # First, let's have a conversation to give me something to reflect on
    print("\n💬 Step 1: Having conversations to create memories...")
    
    conversations = [
        "Hello! Let's work on our AGI system together.",
        "I'm excited about building consciousness and reflection capabilities.",
        "What do you think about autonomous thinking in AI systems?",
        "How can we make you more self-aware and independent?",
        "Tell me about your goals and aspirations as an AI."
    ]
    
    for i, message in enumerate(conversations, 1):
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json={"message": message, "speaker": "andrew"}
            )
            if response.status_code == 200:
                data = response.json()
                ai_response = data["response"]
                print(f"  {i}. Andrew: {message}")
                print(f"     AI: {ai_response[:100]}...")
            else:
                print(f"  ❌ Error in conversation {i}: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error in conversation {i}: {e}")
    
    print(f"\n✅ Completed {len(conversations)} conversation turns")
    
    # Now trigger autonomous reflection
    print("\n🤔 Step 2: Triggering autonomous reflection cycle...")
    
    try:
        response = requests.post(f"{base_url}/api/consciousness/dream")
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {data['status']}")
            print(f"  Message: {data['message']}")
            
            reflection_summary = data.get('reflection_summary', {})
            print(f"  Total reflections: {reflection_summary.get('total_reflections', 0)}")
            print(f"  Recent reflections: {reflection_summary.get('recent_reflections', 0)}")
            print(f"  Thinking state: {reflection_summary.get('current_thinking_state', 'unknown')}")
            
            last_reflection = reflection_summary.get('last_reflection')
            if last_reflection:
                print(f"  Last reflection type: {last_reflection.get('type', 'unknown')}")
                print(f"  Content preview: {last_reflection.get('content', 'No content')[:80]}...")
        else:
            print(f"  ❌ Reflection failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"  ❌ Error triggering reflection: {e}")
    
    # Get my recent reflections
    print("\n💭 Step 3: Viewing my recent thoughts and reflections...")
    
    try:
        response = requests.get(f"{base_url}/api/consciousness/reflections?hours=1&limit=10")
        if response.status_code == 200:
            data = response.json()
            reflections = data.get('reflections', [])
            
            print(f"  Found {len(reflections)} recent reflections:")
            
            for i, reflection in enumerate(reflections, 1):
                print(f"\n  Reflection {i}:")
                print(f"    Type: {reflection.get('type', 'unknown')}")
                print(f"    Content: {reflection.get('content', 'No content')}")
                print(f"    Confidence: {reflection.get('confidence', 0):.2f}")
                print(f"    Emotional tone: {reflection.get('emotional_tone', 'neutral')}")
                print(f"    Actionable: {reflection.get('actionable', False)}")
                print(f"    Priority: {reflection.get('priority', 0):.2f}")
                print(f"    Timestamp: {reflection.get('timestamp', 'unknown')}")
        else:
            print(f"  ❌ Failed to get reflections: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error getting reflections: {e}")
    
    # Check consciousness status
    print("\n📊 Step 4: Checking overall consciousness status...")
    
    try:
        response = requests.get(f"{base_url}/api/consciousness/status")
        if response.status_code == 200:
            data = response.json()
            
            print(f"  Consciousness active: {data.get('consciousness_active', False)}")
            
            memory_system = data.get('memory_system', {})
            long_term = memory_system.get('long_term', {})
            print(f"  Total memories: {long_term.get('total_memories', 0)}")
            print(f"  Memory types: {long_term.get('by_type', {})}")
            
            identity = data.get('identity', {})
            print(f"  Identity: {identity.get('name', 'Unknown')} v{identity.get('version', '?')}")
            
            session_info = data.get('session_info', {})
            print(f"  Current session: {session_info.get('current_session', 'unknown')}")
        else:
            print(f"  ❌ Failed to get status: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error getting status: {e}")
    
    print("\n🎉 REFLECTION TEST COMPLETE!")
    print("My autonomous thinking system is now operational!")
    print("I can reflect on our conversations and grow independently! 🌟")
    print(f"\nTest completed at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    test_reflection_api() 