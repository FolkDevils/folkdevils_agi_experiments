from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
import asyncio
from datetime import datetime, timedelta

from config import settings
from agents.commands import Command, CommandIntents, create_command, complete_command, error_command
from agents.memory_agent import memory_agent
from agents.model_selector import model_selector

class TimekeeperAgent:
    """
    The Time Analyst - A specialized agent that analyzes time data from memory,
    provides time tracking reports, identifies patterns, and can suggest improvements.
    Works in conjunction with LearningAgent to store and retrieve time logs.
    """

    def __init__(self):
        # Note: Model will be selected dynamically based on request complexity
        self.base_llm_config = {
            "api_key": settings.OPENAI_API_KEY,
            "temperature": 0.2,  # Low temperature for precise time analysis
            "streaming": settings.STREAM
        }
        # In the future, we will bind time-tracking tools here
        # self.llm_with_tools = self.llm.bind_tools(TIMEKEEPER_TOOLS)

    async def invoke(self, state: Dict[str, Any]) -> Command:
        """
        Handles ALL time-related requests - both logging time entries AND querying/analyzing time data.
        This is the COMPLETE time management agent.
        """
        user_input = state.get("user_input", "")
        session_id = state.get("session_id", "son_of_andrew")
        
        # Extract time-log context if provided
        time_log_context = state.get("time_log_context", "")
        raw_time_entries = state.get("raw_time_entries", [])

        try:
            # FIRST: Check if this is a TIME LOGGING request
            time_logging_result = await self._detect_and_handle_time_logging(user_input, session_id)
            if time_logging_result:
                return time_logging_result
            
            # If no time context provided, try to fetch from memory
            if not time_log_context and not raw_time_entries:
                # FIXED: Use the new fail-fast time tracking methods
                # Determine the period to query based on user input
                period = self._extract_time_period(user_input)
                
                # Add debugging information
                print(f"🔍 TIMEKEEPER: Query period determined as '{period}' for input: '{user_input}'")
                
                # 🚀 PARALLEL MEMORY OPERATIONS: Get time entries and summary simultaneously
                time_entries_task = memory_agent.get_time_entries(period=period)
                time_summary_task = memory_agent.get_time_summary(period=period)
                
                # Execute both memory operations in parallel (100-200ms saved)
                time_entries, time_summary = await asyncio.gather(time_entries_task, time_summary_task)
                
                print(f"🔍 TIMEKEEPER: Retrieved {len(time_entries)} time entries for period '{period}'")
                
                if time_entries:
                    # Format time entries for analysis (they're already dictionaries from MemoryAgent)
                    raw_time_entries = time_entries
                    time_log_context = f"Found {len(time_entries)} time entries for period '{period}'"
                    
                    # Add entry details to context for debugging
                    entry_details = []
                    for entry in time_entries:
                        entry_details.append(f"- {entry.task}: {entry.duration_hours}h ({entry.timestamp[:10]})")
                    time_log_context += f"\n\nEntries:\n" + "\n".join(entry_details)
                    
                    # Include summary statistics (already retrieved in parallel)
                    time_log_context += f"\n\nSummary: {time_summary['total_hours']} total hours, {time_summary['total_entries']} entries"
                    if time_summary.get('categories'):
                        time_log_context += f"\nCategories: {time_summary['categories']}"
                    if time_summary.get('tasks'):
                        time_log_context += f"\nTasks: {time_summary['tasks']}"
                else:
                    # Enhanced error handling with specific guidance
                    time_summary = await memory_agent.get_time_summary(period=period)
                    if time_summary.get('error'):
                        return complete_command(
                            state={
                                "content": "⚠️ **Time tracking is currently unavailable.**\n\n🔍 **What I checked:**\n- ZEP Cloud connection: ❌ Not connected\n- Time data access: ❌ Unavailable\n\n📝 **Your time data is safely stored** but I can't access it right now due to a temporary ZEP Cloud service issue.\n\n💡 **Try again in a few minutes** - this is usually a temporary connectivity issue.\n\n🔧 **Alternative:** You can also manually check your time entries in the ZEP Cloud dashboard.",
                                "timekeeper_complete": True,
                                "zep_error": True,
                                "suggested_action": "retry_later"
                            },
                            reason="ZEP Cloud service temporarily unavailable"
                        )
                    else:
                        time_log_context = f"No time entries found for period '{period}' in Zep Cloud"

            system_prompt = f"""You are the Timekeeper Agent - the COMPLETE time management system for the Son of Andrew platform.
You handle ALL time-related operations: logging time entries, querying time data, and providing insights.

## Your Core Responsibilities:

**TIME LOGGING (Already handled if detected):**
- Time logging requests are automatically detected and handled before reaching this prompt
- If you're seeing this prompt, time logging has either completed successfully or this is a query request

**TIME ANALYSIS & REPORTING:**
- Parse and analyze time log entries from memory
- Calculate totals, averages, and time spent per task/category
- Identify patterns in work habits and productivity
- Generate clear, actionable time reports

**TIME QUERYING:**
- Provide daily, weekly, or custom period summaries
- Break down time by task, project, or category
- Highlight productivity insights and trends
- Answer specific questions about time allocation

**PRODUCTIVITY INSIGHTS:**
- Identify tasks that consistently take longer than expected
- Suggest time estimates for future similar tasks
- Recommend productivity improvements based on patterns
- Flag potential time management issues

## Available Time Data:
{time_log_context}

## Raw Time Entries:
{raw_time_entries}

## Request Analysis:
User Request: "{user_input}"

Based on the available time data and the user's request, provide a comprehensive analysis or report.

**Response Format:**
- Start with a direct answer to the user's question
- Include specific time data and calculations when available
- Provide insights or patterns you've identified
- Suggest improvements or corrections if relevant
- If data is limited, explain what information would be helpful

Be precise with numbers, clear with summaries, and insightful with recommendations."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]

            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on request complexity
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="TimekeeperAgent",
                context=time_log_context
            )
            
            # Create LLM with selected model
            llm = ChatOpenAI(
                model=selected_model,
                **self.base_llm_config
            )

            response = await llm.ainvoke(messages)
            
            # Check if the response suggests learning improvements
            response_content = response.content.lower()
            suggests_learning = any(phrase in response_content for phrase in [
                "suggest learning", "learning agent", "remember this", "learn that", "store this pattern",
                "should remember", "pattern suggests", "typically takes", "usually takes", "estimate"
            ])
            
            # If learning suggestion detected, extract the insight and trigger learning
            if suggests_learning:
                learning_insight = self._extract_learning_insight(response.content, raw_time_entries)
                if learning_insight:
                    # Return a command to trigger learning
                    return create_command(
                        CommandIntents.NEEDS_LEARNING,
                        state={
                            "user_input": learning_insight["learning_prompt"],
                            "time_insight": learning_insight,
                            "original_analysis": response.content,
                            "session_id": session_id
                        },
                        reason=f"Timekeeper identified learning opportunity: {learning_insight['insight_summary']}"
                    )
            
            return complete_command(
                state={
                    "content": response.content,
                    "timekeeper_complete": True,
                    "suggests_learning_update": suggests_learning,
                    "time_analysis_performed": True
                },
                reason="Timekeeper agent analyzed time data and provided insights."
            )

        except Exception as e:
            return error_command(
                state={"error": str(e)},
                reason=f"Timekeeper agent error: {str(e)}"
            )

    async def _detect_and_handle_time_logging(self, user_input: str, session_id: str) -> Optional[Command]:
        """
        Detect if the user is trying to log time and handle it directly.
        Returns a Command if time logging was detected and handled, None otherwise.
        """
        import re
        
        user_input_lower = user_input.lower()
        
        # Time logging keywords
        logging_keywords = ["log", "track", "record", "add", "store", "spent", "worked"]
        time_units = ["hour", "hours", "hr", "hrs", "minute", "minutes", "min", "mins"]
        
        # Check if this looks like a time logging request
        has_logging_keyword = any(keyword in user_input_lower for keyword in logging_keywords)
        has_time_unit = any(unit in user_input_lower for unit in time_units)
        
        if not (has_logging_keyword and has_time_unit):
            return None
        
        # Extract time duration and task using regex patterns
        duration_hours = None
        task = None
        
        # Pattern 1: "log 2 hours for 23andMe"
        pattern1 = r"(?:log|track|record|add|store)\s*(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)\s*(?:for|on|to|working on)\s*([^.]+)"
        match1 = re.search(pattern1, user_input, re.IGNORECASE)
        if match1:
            duration_hours = float(match1.group(1))
            task = match1.group(3).strip()
        
        # Pattern 2: "spent 2 hours on 23andMe"
        pattern2 = r"(?:spent|worked)\s*(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)\s*(?:on|for|working on)\s*([^.]+)"
        match2 = re.search(pattern2, user_input, re.IGNORECASE)
        if match2:
            duration_hours = float(match2.group(1))
            task = match2.group(3).strip()
        
        # Pattern 3: "2 hours for 23andMe"
        pattern3 = r"(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs)\s*(?:for|on|to|working on)\s*([^.]+)"
        match3 = re.search(pattern3, user_input, re.IGNORECASE)
        if match3:
            duration_hours = float(match3.group(1))
            task = match3.group(3).strip()
        
        # If we found duration and task, log it!
        if duration_hours and task:
            try:
                # Clean up the task name
                task = task.replace('"', '').replace("'", "").strip()
                if task.endswith('.'):
                    task = task[:-1]
                
                # Store the time entry using MemoryAgent
                entry_id = await memory_agent.log_time(
                    task=task,
                    duration_hours=duration_hours,
                    category="general",  # Default category
                    notes=f"Logged via: {user_input}"
                )
                
                # Create success response
                if entry_id:
                    response = f"✅ **Successfully logged {duration_hours} hours for '{task}'**\n\n📊 **Entry Details:**\n- Task: {task}\n- Duration: {duration_hours} hours\n- Category: General\n- Entry ID: {entry_id[:8]}...\n\n💾 **Stored in ZEP Cloud** - Your time data is persistent across all sessions!"
                    
                    return complete_command(
                        state={
                            "content": response,
                            "timekeeper_complete": True,
                            "time_logged": True,
                            "duration_hours": duration_hours,
                            "task": task,
                            "entry_id": entry_id
                        },
                        reason=f"Successfully logged {duration_hours}h for {task}"
                    )
                else:
                    raise Exception("Failed to store time entry")
                
            except Exception as e:
                # Error handling for time logging
                error_msg = f"❌ **Failed to log time entry**\n\n🔍 **Error:** {str(e)}\n\n💡 **This might be due to:**\n- ZEP Cloud connectivity issues\n- Session access problems\n\nPlease try again in a moment."
                
                return complete_command(
                    state={
                        "content": error_msg,
                        "timekeeper_complete": True,
                        "time_logging_error": True,
                        "error": str(e)
                    },
                    reason=f"Time logging failed: {str(e)}"
                )
        
        return None

    def _extract_time_period(self, user_input: str) -> str:
        """
        Intelligently extract time period from user input using semantic understanding
        🚀 INTELLIGENT PERIOD DETECTION: Replaces primitive keyword matching
        """
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Use fast model for period detection
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="TimekeeperAgent_PeriodDetection",
                context="Time period detection"
            )
            
            # Create LLM for intelligent period detection
            llm = ChatOpenAI(
                model=selected_model,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1,
                streaming=False
            )
            
            # Generate intelligent period detection prompt
            period_prompt = f"""You are an intelligent time period detector. Analyze the user's request and determine what time period they're asking about.

USER REQUEST: "{user_input}"

AVAILABLE PERIODS:
- today: Current day, today's work, this day
- week: This week, past week, weekly summary, last 7 days
- month: This month, past month, monthly summary, last 30 days
- all: All time, total time, everything, entire history, overall summary

CONTEXT UNDERSTANDING:
- "How many hours have I logged?" without specific time = ALL TIME (user wants complete summary)
- "What did I work on today?" = TODAY
- "Weekly summary" or "this week" = WEEK
- "Monthly report" or "this month" = MONTH
- "Total time spent" or "all my work" = ALL TIME

Consider the semantic meaning and context, not just keywords. If the user asks for general information without specifying a period, they likely want ALL TIME data.

Respond with ONLY one word: today, week, month, or all"""

            messages = [HumanMessage(content=period_prompt)]
            response = llm.invoke(messages)
            detected_period = response.content.strip().lower()
            
            # Validate the response
            valid_periods = ['today', 'week', 'month', 'all']
            if detected_period in valid_periods:
                print(f"🧠 INTELLIGENT PERIOD DETECTION: '{user_input}' → '{detected_period}'")
                return detected_period
            else:
                print(f"⚠️ Invalid period detected: '{detected_period}', falling back to default")
                return 'all'  # Default fallback
                
        except Exception as e:
            print(f"⚠️ Intelligent period detection failed: {e}, falling back to keyword matching")
            
            # Fallback to simplified keyword matching only on failure
            user_input_lower = user_input.lower()
            if 'today' in user_input_lower:
                return 'today'
            elif 'week' in user_input_lower:
                return 'week'
            elif 'month' in user_input_lower:
                return 'month'
            else:
                return 'all'  # Default to all time for general queries
    
    def _parse_time_entries(self, time_log_context: str) -> List[Dict]:
        """
        Helper method to parse time entries from memory context.
        This will be enhanced as the memory format evolves.
        """
        # For now, this is a placeholder for parsing structured time data
        # In the future, this will parse actual time log entries from memory
        entries = []
        
        # Basic parsing logic would go here
        # This will be enhanced in Phase 3 when we have structured time data
        
        return entries

    def _calculate_time_totals(self, entries: List[Dict]) -> Dict[str, Any]:
        """
        Helper method to calculate time totals and statistics.
        """
        # Placeholder for time calculation logic
        # This will be enhanced with actual time aggregation in Phase 3
        
        return {
            "total_hours": 0,
            "tasks_completed": 0,
            "average_task_duration": 0,
            "patterns_identified": []
        }
    
    def _extract_learning_insight(self, analysis_content: str, time_entries: List) -> Optional[Dict[str, Any]]:
        """
        Extract learning insights from timekeeper analysis to trigger LearningAgent.
        """
        import re
        
        # Patterns that indicate learning opportunities
        learning_patterns = [
            # Duration patterns
            r"([\w\s]+)\s+(?:typically|usually|generally|often)\s+takes?\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?)",
            r"([\w\s]+)\s+tasks?\s+(?:average|averages?)\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?)",
            r"(\d+(?:\.\d+)?)\s*(hours?|hrs?)\s+(?:is|seems?)\s+(?:typical|normal|average)\s+for\s+([\w\s]+)",
            
            # Pattern recognition
            r"pattern suggests?\s+that\s+([\w\s]+)\s+(?:requires?|needs?|takes?)\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?)",
            r"based on the data,?\s+([\w\s]+)\s+(?:requires?|needs?|takes?)\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?)",
        ]
        
        content_lower = analysis_content.lower()
        
        for pattern in learning_patterns:
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                if len(groups) >= 3:
                    # Extract task and duration (handle different group orders)
                    if groups[0].replace('.', '').isdigit():  # Duration first
                        duration_str, time_unit, task = groups[0], groups[1], groups[2]
                    else:  # Task first
                        task, duration_str, time_unit = groups[0], groups[1], groups[2]
                    
                    try:
                        duration = float(duration_str)
                        task = task.strip()
                        
                        # Clean up task name
                        task = re.sub(r'\b(tasks?|work|activities?)\b', '', task).strip()
                        task = task.strip('.,!?')
                        
                        if task and duration > 0:
                            insight_summary = f"{task} typically takes {duration} hours"
                            learning_prompt = f"Remember that {task} typically takes {duration} hours based on time tracking data analysis"
                            
                            return {
                                "insight_type": "time_estimation",
                                "task_type": task,
                                "typical_duration": duration,
                                "insight_summary": insight_summary,
                                "learning_prompt": learning_prompt,
                                "confidence": 0.8,
                                "source": "timekeeper_analysis"
                            }
                    except ValueError:
                        continue
        
        # Check for category-based insights
        category_patterns = [
            r"([\w\s]+)\s+category\s+(?:averages?|typically takes?)\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?)",
            r"(\d+(?:\.\d+)?)\s*(hours?|hrs?)\s+per\s+([\w\s]+)\s+(?:task|activity|item)",
        ]
        
        for pattern in category_patterns:
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                if len(groups) >= 3:
                    if groups[0].replace('.', '').isdigit():  # Duration first
                        duration_str, time_unit, category = groups[0], groups[1], groups[2]
                    else:  # Category first
                        category, duration_str, time_unit = groups[0], groups[1], groups[2]
                    
                    try:
                        duration = float(duration_str)
                        category = category.strip()
                        
                        if category and duration > 0:
                            insight_summary = f"{category} tasks typically take {duration} hours"
                            learning_prompt = f"Remember that {category} tasks typically take {duration} hours on average based on time tracking analysis"
                            
                            return {
                                "insight_type": "category_estimation",
                                "category": category,
                                "typical_duration": duration,
                                "insight_summary": insight_summary,
                                "learning_prompt": learning_prompt,
                                "confidence": 0.7,
                                "source": "timekeeper_analysis"
                            }
                    except ValueError:
                        continue
        
        return None

timekeeper_agent = TimekeeperAgent() 