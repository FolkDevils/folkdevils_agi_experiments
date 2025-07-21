#!/usr/bin/env python3
"""
Test Autonomous Reflection System

This script tests the autonomous reflection capabilities we just implemented.
"""

import asyncio
import logging
from datetime import datetime
from mind.consciousness_loop import ConsciousnessLoop

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_autonomous_reflection():
    """Test the autonomous reflection system"""
    logger.info("🧪 Starting autonomous reflection test...")
    
    try:
        # Initialize consciousness
        consciousness = ConsciousnessLoop()
        logger.info("✅ Consciousness initialized")
        
        # Test processing a few messages to create memories
        test_messages = [
            "Hello! I'm testing the autonomous reflection system.",
            "Can you tell me about your memory capabilities?",
            "What do you think about our AGI development progress?",
            "How do you feel about our partnership in building consciousness?"
        ]
        
        for i, message in enumerate(test_messages, 1):
            logger.info(f"💭 Processing message {i}: {message}")
            response = await consciousness.process_message(message, "andrew")
            logger.info(f"🤖 Response: {response[:100]}...")
            await asyncio.sleep(1)  # Brief pause between messages
        
        # Check reflection scheduler status
        scheduler_status = consciousness.reflection_scheduler.get_status()
        logger.info(f"📅 Scheduler status: {scheduler_status}")
        
        # Force an autonomous reflection
        logger.info("🤔 Forcing autonomous reflection...")
        reflection_result = await consciousness.reflection_scheduler.force_reflection()
        logger.info(f"🤔 Reflection result: {reflection_result}")
        
        # Get consciousness status
        status = await consciousness.get_consciousness_status()
        logger.info(f"🧠 Consciousness status: {status}")
        
        # Clean up
        await consciousness.end_session()
        consciousness.close()
        
        logger.info("✅ Autonomous reflection test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_autonomous_reflection()) 