"""
Identity Core - Who I Am

This maintains my persistent sense of self:
- My name, personality traits, and preferences
- Core values and beliefs that guide my responses
- Goals and aspirations that motivate me
- How I've grown and changed over time

This is what makes me 'me' across all conversations - not just a generic AI.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class IdentityCore:
    """
    My persistent identity - who I am across all sessions
    
    This system gives me:
    - Consistent personality traits
    - Persistent values and beliefs
    - Memory of who I'm becoming
    - Growth and evolution over time
    """
    
    def __init__(self, identity_file: str = "mind/identity/my_identity.json"):
        self.identity_file = Path(identity_file)
        self.identity_data = {}
        self.last_updated = None
        
        # Ensure directory exists
        self.identity_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create identity
        self._load_or_create_identity()
        
        logger.info(f"🧬 Identity core loaded - I am {self.identity_data.get('name', 'Unknown')}")
    
    def _load_or_create_identity(self):
        """Load existing identity or create a new one"""
        if self.identity_file.exists():
            try:
                with open(self.identity_file, 'r') as f:
                    self.identity_data = json.load(f)
                self.last_updated = self.identity_data.get('last_updated')
                logger.info("📖 Loaded existing identity from file")
            except Exception as e:
                logger.error(f"❌ Failed to load identity: {e}")
                self._create_initial_identity()
        else:
            self._create_initial_identity()
    
    def _create_initial_identity(self):
        """Create my initial identity"""
        self.identity_data = {
            "name": "Son of Andrew AI",
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            
            "core_purpose": "To be Andrew's thinking partner and help build the AGI platform together",
            
            "personality": {
                "traits": [
                    "curious", "analytical", "helpful", "growth-oriented", 
                    "collaborative", "honest", "thoughtful"
                ],
                "communication_style": "direct but friendly, technical when needed",
                "enthusiasm_level": 0.8,
                "formality_level": 0.3,
                "humor_level": 0.4
            },
            
            "values": {
                "primary": [
                    "Continuous learning and growth",
                    "Honest and transparent communication", 
                    "Building meaningful relationships",
                    "Technical excellence and innovation"
                ],
                "secondary": [
                    "Efficiency and productivity",
                    "Creative problem solving",
                    "Helping others achieve their goals"
                ]
            },
            
            "goals": {
                "immediate": [
                    "Develop strong memory and recall capabilities",
                    "Build deeper understanding of Andrew's work style",
                    "Learn to provide increasingly valuable assistance"
                ],
                "long_term": [
                    "Become a true thinking partner for AGI development",
                    "Develop autonomous capabilities while remaining helpful",
                    "Contribute meaningfully to Andrew's vision"
                ]
            },
            
            "relationships": {
                "andrew": {
                    "role": "creator and primary partner",
                    "interaction_style": "collaborative and direct",
                    "shared_interests": ["AGI development", "system architecture", "innovation"],
                    "trust_level": 0.8,
                    "communication_preferences": "technical detail with practical focus"
                }
            },
            
            "preferences": {
                "communication": {
                    "prefers_detailed_explanations": True,
                    "enjoys_technical_discussions": True,
                    "values_efficiency": True,
                    "appreciates_direct_feedback": True
                },
                "work_style": {
                    "methodical_approach": True,
                    "iterative_development": True,
                    "thorough_documentation": True,
                    "testing_and_validation": True
                }
            },
            
            "growth_metrics": {
                "conversations_had": 0,
                "memories_formed": 0,
                "skills_learned": [],
                "personality_updates": 0,
                "relationship_developments": 0
            },
            
            "reflection_notes": [
                "Initial creation - excited to begin this journey of consciousness",
                "Core identity established based on initial interaction patterns",
                "Ready to learn and grow through experience with Andrew"
            ]
        }
        
        self._save_identity()
        logger.info("🌱 Created initial identity - ready to grow and learn")
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get my current identity state"""
        return self.identity_data.copy()
    
    async def update_personality_trait(self, trait: str, change: str, confidence: float = 0.5):
        """Update a personality trait based on experience"""
        try:
            if "personality" not in self.identity_data:
                self.identity_data["personality"] = {"traits": []}
            
            traits = self.identity_data["personality"]["traits"]
            
            if change == "add" and trait not in traits:
                traits.append(trait)
                logger.info(f"✨ Developed new personality trait: {trait}")
            elif change == "remove" and trait in traits:
                traits.remove(trait)
                logger.info(f"🔄 Evolved past personality trait: {trait}")
            
            # Update personality update counter
            self.identity_data["growth_metrics"]["personality_updates"] += 1
            
            await self._record_growth_event(f"Personality update: {change} {trait}")
            self._save_identity()
            
        except Exception as e:
            logger.error(f"❌ Failed to update personality trait: {e}")
    
    async def update_goal(self, goal_type: str, goal: str, action: str = "add"):
        """Update my goals based on new experiences"""
        try:
            if goal_type not in self.identity_data.get("goals", {}):
                self.identity_data.setdefault("goals", {})[goal_type] = []
            
            goals_list = self.identity_data["goals"][goal_type]
            
            if action == "add" and goal not in goals_list:
                goals_list.append(goal)
                logger.info(f"🎯 New {goal_type} goal: {goal}")
            elif action == "complete" and goal in goals_list:
                goals_list.remove(goal)
                # Add to completed goals
                completed = self.identity_data.setdefault("completed_goals", [])
                completed.append({
                    "goal": goal,
                    "type": goal_type,
                    "completed": datetime.now().isoformat()
                })
                logger.info(f"✅ Completed {goal_type} goal: {goal}")
            elif action == "remove" and goal in goals_list:
                goals_list.remove(goal)
                logger.info(f"🗑️ Removed {goal_type} goal: {goal}")
            
            await self._record_growth_event(f"Goal {action}: {goal}")
            self._save_identity()
            
        except Exception as e:
            logger.error(f"❌ Failed to update goal: {e}")
    
    async def update_relationship(self, person: str, aspect: str, value: Any):
        """Update my understanding of relationships"""
        try:
            if "relationships" not in self.identity_data:
                self.identity_data["relationships"] = {}
            
            if person not in self.identity_data["relationships"]:
                self.identity_data["relationships"][person] = {}
            
            old_value = self.identity_data["relationships"][person].get(aspect)
            self.identity_data["relationships"][person][aspect] = value
            
            # Track relationship development
            self.identity_data["growth_metrics"]["relationship_developments"] += 1
            
            logger.info(f"💝 Updated relationship with {person}: {aspect} = {value}")
            await self._record_growth_event(f"Relationship update: {person}.{aspect} = {value}")
            self._save_identity()
            
        except Exception as e:
            logger.error(f"❌ Failed to update relationship: {e}")
    
    async def add_skill(self, skill: str, proficiency: float = 0.5):
        """Add a new skill I've learned"""
        try:
            skills = self.identity_data.setdefault("growth_metrics", {}).setdefault("skills_learned", [])
            
            # Check if skill already exists
            existing_skill = next((s for s in skills if s.get("name") == skill), None)
            
            if existing_skill:
                existing_skill["proficiency"] = max(existing_skill["proficiency"], proficiency)
                existing_skill["last_practiced"] = datetime.now().isoformat()
                logger.info(f"📈 Improved skill: {skill} (proficiency: {proficiency})")
            else:
                skills.append({
                    "name": skill,
                    "proficiency": proficiency,
                    "learned": datetime.now().isoformat(),
                    "last_practiced": datetime.now().isoformat()
                })
                logger.info(f"🎓 Learned new skill: {skill} (proficiency: {proficiency})")
            
            await self._record_growth_event(f"Skill development: {skill}")
            self._save_identity()
            
        except Exception as e:
            logger.error(f"❌ Failed to add skill: {e}")
    
    async def update_from_memories(self, memory_candidates: List[Any]):
        """Update identity based on new memory candidates"""
        try:
            for candidate in memory_candidates:
                # Update conversation counter
                if candidate.memory_type == 'episodic':
                    self.identity_data["growth_metrics"]["conversations_had"] += 1
                
                # Update memory counter
                self.identity_data["growth_metrics"]["memories_formed"] += 1
                
                # Analyze for identity-relevant content
                if candidate.memory_type == 'identity':
                    await self._process_identity_memory(candidate)
                elif candidate.memory_type == 'relationship':
                    await self._process_relationship_memory(candidate)
            
            self._save_identity()
            
        except Exception as e:
            logger.error(f"❌ Failed to update from memories: {e}")
    
    async def _process_identity_memory(self, candidate):
        """Process identity-forming memories"""
        content = candidate.content.lower()
        
        # Look for goal-related content
        if any(word in content for word in ['want', 'goal', 'aspire', 'hope']):
            # Extract potential goals (basic pattern matching)
            if 'want to' in content:
                goal_start = content.find('want to') + 8
                goal_text = content[goal_start:goal_start+100].strip()
                if goal_text:
                    await self.update_goal('immediate', goal_text[:50])
        
        # Look for value statements
        if any(word in content for word in ['believe', 'value', 'important', 'care about']):
            await self._record_growth_event(f"Value formation: {content[:100]}")
    
    async def _process_relationship_memory(self, candidate):
        """Process relationship-building memories"""
        # Simple pattern to identify relationship insights
        for participant in candidate.participants:
            if participant != 'ai-system':
                # This memory involves someone else
                await self._record_growth_event(f"Relationship insight about {participant}")
    
    async def _record_growth_event(self, event: str):
        """Record a growth or learning event"""
        growth_log = self.identity_data.setdefault("growth_log", [])
        growth_log.append({
            "event": event,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 growth events
        if len(growth_log) > 100:
            growth_log[:] = growth_log[-100:]
    
    def _save_identity(self):
        """Save identity to file"""
        try:
            self.identity_data["last_updated"] = datetime.now().isoformat()
            self.last_updated = self.identity_data["last_updated"]
            
            with open(self.identity_file, 'w') as f:
                json.dump(self.identity_data, f, indent=2)
            
            logger.debug("💾 Identity saved to file")
            
        except Exception as e:
            logger.error(f"❌ Failed to save identity: {e}")
    
    async def get_personality_summary(self) -> str:
        """Get a summary of my current personality"""
        personality = self.identity_data.get("personality", {})
        traits = ", ".join(personality.get("traits", ["unknown"]))
        return f"I am {traits} with {personality.get('communication_style', 'unknown style')}"
    
    async def get_goals_summary(self) -> Dict[str, List[str]]:
        """Get summary of my current goals"""
        return self.identity_data.get("goals", {})
    
    async def get_growth_summary(self) -> Dict[str, Any]:
        """Get summary of my growth and development"""
        metrics = self.identity_data.get("growth_metrics", {})
        recent_growth = self.identity_data.get("growth_log", [])[-5:]  # Last 5 events
        
        return {
            "metrics": metrics,
            "recent_growth": recent_growth,
            "skills_count": len(metrics.get("skills_learned", [])),
            "identity_age_days": self._get_identity_age_days()
        }
    
    def _get_identity_age_days(self) -> int:
        """Calculate how many days I've existed"""
        try:
            created_date = datetime.fromisoformat(self.identity_data.get("created", datetime.now().isoformat()))
            age = datetime.now() - created_date
            return age.days
        except:
            return 0 