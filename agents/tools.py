import os
import json
import base64
import uuid
import asyncio
import aiofiles
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
from memory_manager import memory_manager

load_dotenv()

# Initialize the LLM for vision analysis
vision_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.4
)

@tool
def analyze_image(base64_image: str, analysis_type: str = "general", context: str = "") -> str:
    """
    Analyze an image and provide detailed descriptions.
    
    Args:
        base64_image: Base64 encoded image data
        analysis_type: Type of analysis - "general", "conversation", "performance", or "content"
        context: Additional context for the analysis
    
    Returns:
        Detailed analysis of the image
    """
    
    prompt_templates = {
        "general": "Analyze this image in detail. Describe what you see, including objects, people, setting, colors, mood, and any notable details.",
        "conversation": "Look at this image in the context of a conversation. What's happening? What might this be related to? What questions or topics might this spark?",
        "performance": "Analyze this image for performance evaluation context. What does this image suggest about communication effectiveness, engagement, or outcomes?",
        "content": "Analyze this image for content creation. What story does it tell? What themes, emotions, or messages does it convey?"
    }
    
    base_prompt = prompt_templates.get(analysis_type, prompt_templates["general"])
    
    if context:
        base_prompt += f"\n\nAdditional context: {context}"
    
    base_prompt += "\n\nProvide a clear, detailed analysis. Focus on facts and observable details."
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": base_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            }
        ]
    )
    
    response = vision_llm.invoke([message])
    return response.content

@tool
async def get_recent_conversations(limit: int = 5, hours_back: int = 24) -> str:
    """
    Get recent conversations from storage.
    
    Args:
        limit: Maximum number of conversations to return
        hours_back: How many hours back to look for conversations
    
    Returns:
        JSON string of recent conversations
    """
    storage_path = "conversations.json"
    
    # Ensure storage exists
    if not os.path.exists(storage_path):
        async with aiofiles.open(storage_path, 'w') as f:
            await f.write(json.dumps({"conversations": []}))
        return json.dumps({"conversations": [], "count": 0})
    
    try:
        async with aiofiles.open(storage_path, 'r') as f:
            content = await f.read()
            data = json.loads(content)
        
        conversations = data.get("conversations", [])
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter by time and limit
        recent_conversations = []
        for conv in conversations:
            conv_time = datetime.fromisoformat(conv.get("timestamp", "2000-01-01T00:00:00"))
            if conv_time > cutoff_time:
                recent_conversations.append(conv)
        
        # Sort by most recent first and limit
        recent_conversations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        recent_conversations = recent_conversations[:limit]
        
        return json.dumps({
            "conversations": recent_conversations,
            "count": len(recent_conversations),
            "time_range_hours": hours_back
        })
    
    except Exception as e:
        return json.dumps({"error": f"Failed to retrieve conversations: {str(e)}"})

@tool
async def store_conversation(conversation: str, topic: str = "", context: str = "", performance_note: str = "") -> str:
    """
    Store a conversation with metadata.
    
    Args:
        conversation: The conversation content to store
        topic: Optional topic classification
        context: Optional context information
        performance_note: Optional performance evaluation note
    
    Returns:
        JSON string with storage result
    """
    storage_path = "conversations.json"
    
    # Ensure storage exists
    if not os.path.exists(storage_path):
        async with aiofiles.open(storage_path, 'w') as f:
            await f.write(json.dumps({"conversations": []}))
    
    try:
        async with aiofiles.open(storage_path, 'r') as f:
            content = await f.read()
            data = json.loads(content)
        
        conversation_data = {
            "content": conversation,
            "topic": topic,
            "context": context,
            "performance_note": performance_note,
            "timestamp": datetime.now().isoformat(),
            "id": f"conv_{int(datetime.now().timestamp())}"
        }
        
        data["conversations"].append(conversation_data)
        
        # Keep only last 100 conversations
        data["conversations"] = data["conversations"][-100:]
        
        async with aiofiles.open(storage_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))
        
        return json.dumps({"stored": True, "id": conversation_data["id"]})
    
    except Exception as e:
        return json.dumps({"error": f"Failed to store conversation: {str(e)}"})

@tool
async def get_conversation_stats(hours_back: int = 24) -> str:
    """
    Get statistics about recent conversations.
    
    Args:
        hours_back: How many hours back to analyze
    
    Returns:
        JSON string with conversation statistics
    """
    storage_path = "conversations.json"
    
    if not os.path.exists(storage_path):
        return json.dumps({
            "total_conversations": 0,
            "time_range_hours": hours_back,
            "topics": [],
            "avg_length": 0,
            "performance_notes": []
        })
    
    try:
        async with aiofiles.open(storage_path, 'r') as f:
            content = await f.read()
            data = json.loads(content)
        
        conversations = data.get("conversations", [])
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter recent conversations
        recent_conversations = []
        for conv in conversations:
            conv_time = datetime.fromisoformat(conv.get("timestamp", "2000-01-01T00:00:00"))
            if conv_time > cutoff_time:
                recent_conversations.append(conv)
        
        stats = {
            "total_conversations": len(recent_conversations),
            "time_range_hours": hours_back,
            "topics": [],
            "avg_length": 0,
            "performance_notes": []
        }
        
        if recent_conversations:
            # Calculate average length
            stats["avg_length"] = sum(len(c.get("content", "")) for c in recent_conversations) / len(recent_conversations)
            
            # Aggregate topics
            topic_counts = {}
            for c in recent_conversations:
                topic = c.get("topic")
                if topic:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
            stats["topics"] = [{"topic": t, "count": c} for t, c in topic_counts.items()]
            
            # Get performance notes
            stats["performance_notes"] = [c.get("performance_note") for c in recent_conversations if c.get("performance_note")]

        return json.dumps(stats)
    
    except Exception as e:
        return json.dumps({"error": f"Failed to get conversation stats: {str(e)}"})

# List of all available tools
@tool 
async def get_memory_context(query: str, session_id: str = None, max_messages: int = 5) -> str:
    """
    Get relevant context from previous conversations in this session.
    
    Args:
        query: The current query or topic to find relevant context for
        session_id: Optional session ID to get context from
        max_messages: Maximum number of previous messages to include
    
    Returns:
        String containing relevant conversation context
    """
    try:
        context = await memory_manager.get_memory_context(session_id or "", query)
        return context if context else "No relevant previous context found"
    except Exception as e:
        return f"Error retrieving memory context: {str(e)}"

@tool
async def get_session_summary(session_id: str = None) -> str:
    """
    Get a summary of the current or specified session.
    
    Args:
        session_id: Optional session ID to summarize (defaults to current session)
    
    Returns:
        JSON string containing session summary with workflow patterns and voice compliance
    """
    try:
        summary = await memory_manager.get_session_summary(session_id)
        return json.dumps(summary, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Could not get session summary: {str(e)}"})

# --- Timekeeper Agent Tools ---

TIME_LOG_FILE = "time_log.json"
_time_log_lock = asyncio.Lock()

@tool
async def log_time(task_description: str, duration_minutes: int) -> str:
    """
    Logs time spent on a specific task.

    Args:
        task_description: A description of the task that was worked on.
        duration_minutes: The duration in minutes that was spent on the task.
    
    Returns:
        A JSON string confirming the time was logged.
    """
    async with _time_log_lock:
        try:
            # Ensure the file exists
            if not os.path.exists(TIME_LOG_FILE):
                async with aiofiles.open(TIME_LOG_FILE, 'w') as f:
                    await f.write(json.dumps({"time_entries": []}))

            # Read current entries
            async with aiofiles.open(TIME_LOG_FILE, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            # Add new entry
            entry = {
                "task": task_description,
                "duration_minutes": duration_minutes,
                "timestamp": datetime.now().isoformat(),
                "id": f"time_{int(datetime.now().timestamp())}"
            }
            data["time_entries"].append(entry)

            # Write back to file
            async with aiofiles.open(TIME_LOG_FILE, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            
            return json.dumps({"status": "success", "message": f"Logged {duration_minutes} minutes for '{task_description}'."})

        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

@tool
async def retrieve_time_summary(period_hours: int = 24) -> str:
    """
    Retrieves a summary of time logged within a specific period.

    Args:
        period_hours: The number of hours to look back for time entries.
    
    Returns:
        A JSON string summarizing the time logged, including total time and time per task.
    """
    async with _time_log_lock:
        try:
            if not os.path.exists(TIME_LOG_FILE):
                return json.dumps({"total_minutes": 0, "tasks": {}, "message": "No time has been logged yet."})

            async with aiofiles.open(TIME_LOG_FILE, 'r') as f:
                data = json.loads(await f.read())
            
            entries = data.get("time_entries", [])
            cutoff = datetime.now() - timedelta(hours=period_hours)
            
            recent_entries = [e for e in entries if datetime.fromisoformat(e["timestamp"]) > cutoff]
            
            total_minutes = sum(e["duration_minutes"] for e in recent_entries)
            
            task_summary = {}
            for entry in recent_entries:
                task = entry["task"]
                task_summary[task] = task_summary.get(task, 0) + entry["duration_minutes"]

            return json.dumps({
                "total_minutes": total_minutes,
                "total_hours": round(total_minutes / 60, 2),
                "tasks": task_summary,
                "period_hours": period_hours
            })
            
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})


# --- Memory Tools (Legacy - to be updated/removed) ---
# Keep these for now to avoid breaking existing agents that might call them directly
# In the new architecture, agents should interact with the MemoryManager directly.

AVAILABLE_TOOLS = [
    analyze_image,
    get_recent_conversations, 
    store_conversation,
    get_conversation_stats,
    get_memory_context,
    get_session_summary,
    log_time,
    retrieve_time_summary
] 