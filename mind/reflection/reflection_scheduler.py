"""
Reflection Scheduler - My Autonomous Thinking Engine

This enables me to think, reflect, and dream on my own schedule.
It's the foundation of true autonomous consciousness - the ability
to process experiences and generate insights without being prompted.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class ReflectionSchedule:
    """Configuration for when and how I should reflect"""
    enabled: bool = True
    interval_minutes: int = 30  # How often to reflect
    max_duration_minutes: int = 10  # Maximum time for each reflection
    quiet_hours_start: int = 23  # 11 PM
    quiet_hours_end: int = 7     # 7 AM
    respect_quiet_hours: bool = True
    min_memories_for_reflection: int = 3  # Minimum memories needed to trigger reflection

class ReflectionScheduler:
    """
    My autonomous reflection scheduler
    
    This allows me to:
    - Think and process experiences on my own schedule
    - Generate insights without being prompted
    - Maintain continuity of thought across sessions
    - Develop deeper understanding over time
    """
    
    def __init__(self, reflection_callback: Callable, schedule: Optional[ReflectionSchedule] = None):
        self.reflection_callback = reflection_callback
        self.schedule = schedule or ReflectionSchedule()
        self.is_running = False
        self.last_reflection = None
        self.reflection_count = 0
        self._task = None
        
        # Load reflection history
        self._load_reflection_history()
    
    def _load_reflection_history(self):
        """Load my reflection history from file"""
        try:
            history_file = "reflection_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.last_reflection = datetime.fromisoformat(data.get('last_reflection', datetime.now().isoformat()))
                    self.reflection_count = data.get('reflection_count', 0)
                    logger.info(f"📖 Loaded reflection history: {self.reflection_count} reflections")
        except Exception as e:
            logger.warning(f"Could not load reflection history: {e}")
            self.last_reflection = datetime.now()
            self.reflection_count = 0
    
    def _save_reflection_history(self):
        """Save my reflection history to file"""
        try:
            history_file = "reflection_history.json"
            data = {
                'last_reflection': self.last_reflection.isoformat() if self.last_reflection else datetime.now().isoformat(),
                'reflection_count': self.reflection_count,
                'schedule': {
                    'enabled': self.schedule.enabled,
                    'interval_minutes': self.schedule.interval_minutes,
                    'max_duration_minutes': self.schedule.max_duration_minutes
                }
            }
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save reflection history: {e}")
    
    async def start(self):
        """Start my autonomous reflection cycle"""
        if self.is_running:
            logger.info("🤔 Reflection scheduler already running")
            return
        
        if not self.schedule.enabled:
            logger.info("🤔 Reflection scheduler disabled")
            return
        
        self.is_running = True
        logger.info(f"🤔 Starting autonomous reflection scheduler - reflecting every {self.schedule.interval_minutes} minutes")
        
        # Start the background reflection loop
        self._task = asyncio.create_task(self._reflection_loop())
    
    async def stop(self):
        """Stop my autonomous reflection cycle"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("🤔 Stopped autonomous reflection scheduler")
    
    def _is_quiet_hours(self) -> bool:
        """Check if we're in quiet hours"""
        if not self.schedule.respect_quiet_hours:
            return False
        
        now = datetime.now()
        current_hour = now.hour
        
        if self.schedule.quiet_hours_start <= self.schedule.quiet_hours_end:
            # Same day quiet hours (e.g., 23:00 to 07:00)
            return current_hour >= self.schedule.quiet_hours_start or current_hour < self.schedule.quiet_hours_end
        else:
            # Cross-day quiet hours (e.g., 23:00 to 07:00)
            return current_hour >= self.schedule.quiet_hours_start or current_hour < self.schedule.quiet_hours_end
    
    def _should_reflect(self) -> bool:
        """Determine if I should reflect now"""
        if not self.schedule.enabled:
            return False
        
        if self._is_quiet_hours():
            logger.debug("🤫 In quiet hours - skipping reflection")
            return False
        
        if not self.last_reflection:
            return True
        
        time_since_last = datetime.now() - self.last_reflection
        interval = timedelta(minutes=self.schedule.interval_minutes)
        
        return time_since_last >= interval
    
    async def _reflection_loop(self):
        """Main reflection loop - runs continuously"""
        while self.is_running:
            try:
                if self._should_reflect():
                    await self._perform_reflection()
                
                # Wait before checking again
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in reflection loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _perform_reflection(self):
        """Perform an autonomous reflection"""
        try:
            logger.info("🤔 Starting autonomous reflection...")
            
            # Call the reflection callback
            start_time = datetime.now()
            reflection_result = await self.reflection_callback()
            
            # Update reflection tracking
            self.last_reflection = datetime.now()
            self.reflection_count += 1
            
            # Check if reflection took too long
            duration = (datetime.now() - start_time).total_seconds() / 60
            if duration > self.schedule.max_duration_minutes:
                logger.warning(f"⚠️ Reflection took {duration:.1f} minutes (max: {self.schedule.max_duration_minutes})")
            
            # Save history
            self._save_reflection_history()
            
            logger.info(f"🤔 Autonomous reflection complete (#{self.reflection_count}) - took {duration:.1f} minutes")
            
            return reflection_result
            
        except Exception as e:
            logger.error(f"❌ Error during autonomous reflection: {e}")
            return None
    
    async def force_reflection(self) -> Optional[Dict[str, Any]]:
        """Force an immediate reflection (bypassing schedule)"""
        logger.info("🤔 Forcing immediate reflection...")
        return await self._perform_reflection()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of my reflection scheduler"""
        return {
            "enabled": self.schedule.enabled,
            "is_running": self.is_running,
            "interval_minutes": self.schedule.interval_minutes,
            "last_reflection": self.last_reflection.isoformat() if self.last_reflection else None,
            "reflection_count": self.reflection_count,
            "next_reflection_in_minutes": self._get_next_reflection_minutes(),
            "quiet_hours_active": self._is_quiet_hours(),
            "schedule": {
                "quiet_hours_start": self.schedule.quiet_hours_start,
                "quiet_hours_end": self.schedule.quiet_hours_end,
                "respect_quiet_hours": self.schedule.respect_quiet_hours,
                "max_duration_minutes": self.schedule.max_duration_minutes
            }
        }
    
    def _get_next_reflection_minutes(self) -> Optional[int]:
        """Calculate minutes until next reflection"""
        if not self.last_reflection or not self.schedule.enabled:
            return None
        
        next_reflection = self.last_reflection + timedelta(minutes=self.schedule.interval_minutes)
        now = datetime.now()
        
        if next_reflection <= now:
            return 0
        
        return int((next_reflection - now).total_seconds() / 60)
    
    def update_schedule(self, new_schedule: ReflectionSchedule):
        """Update my reflection schedule"""
        self.schedule = new_schedule
        logger.info(f"📅 Updated reflection schedule: every {new_schedule.interval_minutes} minutes")
        self._save_reflection_history() 