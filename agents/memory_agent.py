"""
MemoryAgent - Clean Interface for Persistent Memory Access

Provides a standardized, semantic interface for agents to access persistent memory
without needing to know about Zep sessions, memory_manager internals, or storage details.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryAgent:
    """
    Clean interface for persistent memory operations.
    
    This wraps the existing memory_manager to provide:
    - Semantic method names
    - Consistent error handling  
    - Simplified agent interfaces
    - Future-proofing for memory system changes
    """
    
    def __init__(self):
        self._memory_manager = None
    
    async def _get_memory_manager(self):
        """Lazy import to avoid circular dependencies"""
        if self._memory_manager is None:
            from memory_manager import memory_manager
            self._memory_manager = memory_manager
        return self._memory_manager
    
    # === FACT MANAGEMENT ===
    
    async def save_fact(self, fact: str, category: str = "general") -> bool:
        """
        Save a persistent fact about Andrew.
        
        Args:
            fact: The fact to store (e.g., "Sarah is my design partner")
            category: Category for organization (e.g., "personal", "work", "preferences")
            
        Returns:
            bool: True if successfully stored
        """
        try:
            memory_manager = await self._get_memory_manager()
            result = await memory_manager.store_persistent_fact(fact, category)
            logger.info(f"💾 MemoryAgent saved fact: {fact[:50]}... (category: {category})")
            return result
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to save fact: {e}")
            return False
    
    async def search_facts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search through stored facts.
        
        Args:
            query: Search query (e.g., "Sarah", "design partner")
            limit: Maximum number of results
            
        Returns:
            List of matching facts with metadata
        """
        try:
            memory_manager = await self._get_memory_manager()
            results = await memory_manager.search_memories(
                query=query, 
                limit=limit, 
                session_id=None  # Search across persistent sessions
            )
            logger.info(f"🔍 MemoryAgent found {len(results)} facts for query: {query}")
            return results
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to search facts: {e}")
            return []
    
    async def get_facts_about(self, entity: str) -> List[Dict[str, Any]]:
        """
        Get all facts related to a specific entity (person, company, etc.).
        
        Args:
            entity: The entity to search for (e.g., "Sarah", "Folk Devils")
            
        Returns:
            List of relevant facts
        """
        return await self.search_facts(entity, limit=20)
    
    # === ADVANCED FACT MANAGEMENT ===
    
    async def forget_fact(self, fact_query: str) -> Dict[str, Any]:
        """
        Delete a fact from memory.
        
        Args:
            fact_query: Query to find the fact to delete (e.g., "Sarah is my partner")
            
        Returns:
            Dict with success status, deleted facts count, and details
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # First, find matching facts
            matching_facts = await self.search_facts(fact_query, limit=10)
            
            if not matching_facts:
                logger.info(f"🗑️ MemoryAgent: No facts found to forget for query: {fact_query}")
                return {
                    "success": False,
                    "message": "No matching facts found to delete",
                    "deleted_count": 0,
                    "facts_found": []
                }
            
            # For now, we'll mark facts as deleted by storing a deletion record
            # This preserves history while making facts inaccessible
            deleted_count = 0
            deleted_facts = []
            
            for fact in matching_facts:
                fact_content = fact.get("content", "")
                fact_id = fact.get("uuid", "") or fact.get("id", "")
                
                # Store deletion record
                deletion_record = f"DELETED_FACT: {fact_content} (deleted on {datetime.now().isoformat()})"
                await memory_manager.store_persistent_fact(deletion_record, "system_deletions")
                
                deleted_facts.append({
                    "content": fact_content,
                    "id": fact_id,
                    "deleted_at": datetime.now().isoformat()
                })
                deleted_count += 1
            
            logger.info(f"🗑️ MemoryAgent forgot {deleted_count} facts for query: {fact_query}")
            
            return {
                "success": True,
                "message": f"Successfully forgot {deleted_count} facts",
                "deleted_count": deleted_count,
                "deleted_facts": deleted_facts
            }
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to forget fact: {e}")
            return {
                "success": False,
                "message": f"Error forgetting fact: {str(e)}",
                "deleted_count": 0,
                "facts_found": []
            }
    
    async def update_fact(self, old_fact_query: str, new_fact: str, category: str = "general") -> Dict[str, Any]:
        """
        Update an existing fact with new information.
        
        Args:
            old_fact_query: Query to find the fact to update
            new_fact: The updated fact content
            category: Category for the new fact
            
        Returns:
            Dict with success status and update details
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Find the existing fact
            matching_facts = await self.search_facts(old_fact_query, limit=5)
            
            if not matching_facts:
                logger.info(f"📝 MemoryAgent: No facts found to update for query: {old_fact_query}")
                return {
                    "success": False,
                    "message": "No matching facts found to update",
                    "old_fact": None,
                    "new_fact": None
                }
            
            # Take the most relevant fact (first result)
            old_fact = matching_facts[0]
            old_content = old_fact.get("content", "")
            
            # Store update history
            update_record = f"FACT_UPDATE: '{old_content}' → '{new_fact}' (updated on {datetime.now().isoformat()})"
            await memory_manager.store_persistent_fact(update_record, "system_updates")
            
            # Store the new fact
            await self.save_fact(new_fact, category)
            
            # Mark old fact as replaced
            replacement_record = f"REPLACED_FACT: {old_content} (replaced on {datetime.now().isoformat()})"
            await memory_manager.store_persistent_fact(replacement_record, "system_replacements")
            
            logger.info(f"📝 MemoryAgent updated fact: '{old_content}' → '{new_fact}'")
            
            return {
                "success": True,
                "message": "Successfully updated fact",
                "old_fact": old_content,
                "new_fact": new_fact,
                "updated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to update fact: {e}")
            return {
                "success": False,
                "message": f"Error updating fact: {str(e)}",
                "old_fact": None,
                "new_fact": None
            }
    
    async def find_similar_facts(self, fact: str, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find facts that might be duplicates or conflicts with the given fact.
        
        Args:
            fact: The fact to compare against
            similarity_threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar facts with similarity scores
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Extract key terms from the fact for searching
            key_terms = self._extract_key_terms(fact)
            search_query = " ".join(key_terms)
            
            # Search for potentially similar facts
            all_facts = await self.search_facts(search_query, limit=20)
            
            similar_facts = []
            for existing_fact in all_facts:
                existing_content = existing_fact.get("content", "")
                
                # Skip system records (deletions, updates, etc.)
                if existing_content.startswith(("DELETED_FACT:", "FACT_UPDATE:", "REPLACED_FACT:")):
                    continue
                
                # Calculate similarity (simple keyword-based for now)
                similarity = self._calculate_similarity(fact, existing_content)
                
                if similarity >= similarity_threshold:
                    similar_facts.append({
                        **existing_fact,
                        "similarity_score": similarity,
                        "comparison_fact": fact
                    })
            
            # Sort by similarity score (highest first)
            similar_facts.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            logger.info(f"🔍 MemoryAgent found {len(similar_facts)} similar facts for: {fact[:50]}...")
            return similar_facts
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to find similar facts: {e}")
            return []
    
    async def detect_conflicts(self, new_fact: str) -> List[Dict[str, Any]]:
        """
        Detect if a new fact conflicts with existing facts.
        
        Args:
            new_fact: The new fact to check for conflicts
            
        Returns:
            List of conflicting facts with conflict analysis
        """
        try:
            # Find similar facts first
            similar_facts = await self.find_similar_facts(new_fact, similarity_threshold=0.5)
            
            conflicts = []
            for fact in similar_facts:
                existing_content = fact.get("content", "")
                
                # Check for contradictory indicators
                if self._are_facts_conflicting(new_fact, existing_content):
                    conflicts.append({
                        **fact,
                        "conflict_type": self._classify_conflict(new_fact, existing_content),
                        "new_fact": new_fact,
                        "confidence": "medium"  # Could be enhanced with ML
                    })
            
            logger.info(f"⚠️ MemoryAgent detected {len(conflicts)} conflicts for: {new_fact[:50]}...")
            return conflicts
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to detect conflicts: {e}")
            return []
    
    async def get_fact_history(self, fact_query: str) -> List[Dict[str, Any]]:
        """
        Get the history of changes for facts matching the query.
        
        Args:
            fact_query: Query to find fact history
            
        Returns:
            List of historical changes (updates, deletions, etc.)
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Search for system records related to this fact
            history_records = []
            
            # Search for updates
            update_query = f"FACT_UPDATE {fact_query}"
            updates = await memory_manager.search_memories(update_query, limit=20)
            for update in updates:
                content = update.get("content", "")
                if content.startswith("FACT_UPDATE:"):
                    history_records.append({
                        **update,
                        "action": "update",
                        "parsed_content": content.replace("FACT_UPDATE: ", "")
                    })
            
            # Search for deletions
            deletion_query = f"DELETED_FACT {fact_query}"
            deletions = await memory_manager.search_memories(deletion_query, limit=20)
            for deletion in deletions:
                content = deletion.get("content", "")
                if content.startswith("DELETED_FACT:"):
                    history_records.append({
                        **deletion,
                        "action": "deletion",
                        "parsed_content": content.replace("DELETED_FACT: ", "")
                    })
            
            # Search for replacements
            replacement_query = f"REPLACED_FACT {fact_query}"
            replacements = await memory_manager.search_memories(replacement_query, limit=20)
            for replacement in replacements:
                content = replacement.get("content", "")
                if content.startswith("REPLACED_FACT:"):
                    history_records.append({
                        **replacement,
                        "action": "replacement",
                        "parsed_content": content.replace("REPLACED_FACT: ", "")
                    })
            
            # Sort by timestamp (most recent first)
            history_records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            logger.info(f"📚 MemoryAgent found {len(history_records)} history records for: {fact_query}")
            return history_records
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to get fact history: {e}")
            return []
    
    # === HELPER METHODS ===
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for similarity comparison"""
        import re
        
        # Remove common words and extract meaningful terms
        stop_words = {'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Split into words and filter
        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms[:10]  # Limit to top 10 terms
    
    def _calculate_similarity(self, fact1: str, fact2: str) -> float:
        """Calculate similarity between two facts (simple keyword-based)"""
        terms1 = set(self._extract_key_terms(fact1))
        terms2 = set(self._extract_key_terms(fact2))
        
        if not terms1 or not terms2:
            return 0.0
        
        intersection = terms1.intersection(terms2)
        union = terms1.union(terms2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _are_facts_conflicting(self, fact1: str, fact2: str) -> bool:
        """Check if two facts are contradictory"""
        fact1_lower = fact1.lower()
        fact2_lower = fact2.lower()
        
        # Check for obvious contradictions
        contradiction_patterns = [
            ("is not", "is"),
            ("was not", "was"),
            ("doesn't", "does"),
            ("isn't", "is"),
            ("wasn't", "was"),
            ("no longer", "still"),
            ("former", "current"),
            ("ex-", "current")
        ]
        
        for negative, positive in contradiction_patterns:
            if (negative in fact1_lower and positive in fact2_lower) or \
               (positive in fact1_lower and negative in fact2_lower):
                return True
        
        return False
    
    def _classify_conflict(self, fact1: str, fact2: str) -> str:
        """Classify the type of conflict between facts"""
        if "is not" in fact1.lower() or "is not" in fact2.lower():
            return "negation"
        elif "was" in fact1.lower() and "is" in fact2.lower():
            return "temporal"
        elif "former" in fact1.lower() or "ex-" in fact1.lower():
            return "status_change"
        else:
            return "general_contradiction"
    
    # === PREFERENCE MANAGEMENT ===
    
    async def save_preference(self, preference: str, category: str = "general") -> bool:
        """
        Save a preference about Andrew's communication style or behavior.
        
        Args:
            preference: The preference to store (e.g., "prefers concise communication")
            category: Category (e.g., "style", "communication", "workflow")
            
        Returns:
            bool: True if successfully stored
        """
        try:
            memory_manager = await self._get_memory_manager()
            result = await memory_manager.store_persistent_preference(preference, category)
            logger.info(f"⚙️ MemoryAgent saved preference: {preference[:50]}... (category: {category})")
            return result
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to save preference: {e}")
            return False
    
    async def get_preferences(self, category: str = None) -> List[Dict[str, Any]]:
        """
        Get stored preferences, optionally filtered by category.
        
        Args:
            category: Optional category filter (e.g., "style", "communication")
            
        Returns:
            List of matching preferences
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Search for preferences with category filter if provided
            query = f"preference {category}" if category else "preference"
            results = await memory_manager.search_memories(
                query=query,
                limit=50,
                session_id=None
            )
            
            # Filter for preference-type entries
            preferences = [
                result for result in results 
                if result.get("content", "").startswith("PREFERENCE:")
            ]
            
            logger.info(f"⚙️ MemoryAgent found {len(preferences)} preferences for category: {category}")
            return preferences
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to get preferences: {e}")
            return []
    
    # === TIME TRACKING ===
    
    async def log_time(self, task: str, duration_hours: float, category: str = "general", notes: str = "") -> Optional[str]:
        """
        Log time spent on a task.
        
        Args:
            task: Description of what was worked on
            duration_hours: Hours spent (can be decimal)
            category: Task category for organization
            notes: Optional additional notes
            
        Returns:
            str: Entry ID if successful, None if failed
        """
        try:
            memory_manager = await self._get_memory_manager()
            entry_id = await memory_manager.store_time_entry(
                session_id="andrew_time_tracking",  # Always use persistent session
                task=task,
                duration_hours=duration_hours,
                category=category,
                notes=notes
            )
            logger.info(f"⏰ MemoryAgent logged time: {task} ({duration_hours}h)")
            return entry_id
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to log time: {e}")
            return None
    
    async def get_time_entries(self, period: str = "today", category: str = None) -> List[Dict[str, Any]]:
        """
        Get time entries for a specific period.
        
        Args:
            period: "today", "week", "month", or "all"
            category: Optional category filter
            
        Returns:
            List of time entries
        """
        try:
            memory_manager = await self._get_memory_manager()
            entries = await memory_manager.get_time_entries(
                session_id="andrew_time_tracking",
                period=period,
                category=category
            )
            logger.info(f"⏰ MemoryAgent retrieved {len(entries)} time entries for {period}")
            return entries
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to get time entries: {e}")
            return []
    
    async def get_time_summary(self, period: str = "today") -> Dict[str, Any]:
        """
        Get a summary of time spent for a period.
        
        Args:
            period: "today", "week", "month", or "all"
            
        Returns:
            Dictionary with time summary statistics
        """
        try:
            memory_manager = await self._get_memory_manager()
            summary = await memory_manager.get_time_summary(
                session_id="andrew_time_tracking",
                period=period
            )
            logger.info(f"⏰ MemoryAgent generated time summary for {period}")
            return summary
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to get time summary: {e}")
            return {}
    
    # === TASK MANAGEMENT ===
    
    async def save_task(self, task: str, priority: str = "medium", due_date: str = None, project: str = None, notes: str = "") -> str:
        """
        🚀 NATURAL TASK STORAGE: Save tasks as natural language
        
        This is the RIGHT WAY - store tasks how humans naturally think about them.
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Create natural task description
            natural_task = f"I need to {task.lower()}"
            
            # Add priority context naturally
            if priority == "urgent":
                natural_task += " urgently"
            elif priority == "high":
                natural_task += " - this is high priority"
            elif priority == "low":
                natural_task += " when I have time"
            
            # Add due date naturally
            if due_date:
                natural_task += f" by {due_date}"
            
            # Add project context naturally
            if project:
                natural_task += f" for the {project} project"
            
            # Add notes naturally
            if notes:
                natural_task += f". {notes}"
            
            # Store as natural language
            success = await memory_manager.store_persistent_fact(natural_task, "tasks")
            
            if success:
                logger.info(f"✅ MemoryAgent saved natural task: {natural_task}")
                return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to save natural task: {e}")
            return None
    
    async def get_tasks(self, status: str = None, priority: str = None, project: str = None) -> List[Dict[str, Any]]:
        """
        🚀 SEMANTIC TASK RETRIEVAL: Find tasks using natural language understanding
        
        This is the RIGHT WAY - use semantic search to find naturally stored tasks.
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Build semantic search queries based on natural language
            search_queries = []
            
            # Base query for tasks
            search_queries.append("I need to")
            search_queries.append("task")
            search_queries.append("to do")
            
            # Add priority-based queries
            if priority:
                if priority == "urgent":
                    search_queries.append("urgently")
                elif priority == "high":
                    search_queries.append("high priority")
                elif priority == "low":
                    search_queries.append("when I have time")
            
            # Add project-based queries
            if project:
                search_queries.append(f"{project} project")
            
            # Add status-based queries (though tasks are naturally pending unless marked done)
            if status == "completed":
                search_queries.append("completed")
                search_queries.append("done")
                search_queries.append("finished")
            
            # Execute semantic search
            all_results = []
            seen_tasks = set()
            
            for search_query in search_queries:
                try:
                    results = await memory_manager.search_memories(search_query, limit=20)
                    for result in results:
                        content = result.get("content", "")
                        # Look for task-like content naturally
                        if ("I need to" in content or 
                            "task" in content.lower() or 
                            "priority" in content.lower() or
                            "project" in content.lower() or
                            "by " in content):  # due dates
                            
                            task_info = self._parse_natural_task_content(content)
                            if task_info and task_info["task"] not in seen_tasks:
                                task_info.update({
                                    "id": result.get("uuid", ""),
                                    "timestamp": result.get("timestamp", ""),
                                    "relevance_score": 1.0
                                })
                                all_results.append(task_info)
                                seen_tasks.add(task_info["task"])
                except Exception as e:
                    logger.warning(f"⚠️ Task search query failed: {search_query} - {e}")
            
            # Sort by priority and relevance
            priority_order = {"urgent": 4, "high": 3, "medium": 2, "low": 1}
            all_results.sort(key=lambda x: (
                priority_order.get(x.get("priority", "medium"), 2),
                x.get("relevance_score", 0)
            ), reverse=True)
            
            logger.info(f"🧠 SEMANTIC TASK RETRIEVAL: Found {len(all_results)} tasks")
            return all_results
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to get tasks semantically: {e}")
            return []
    
    async def complete_task(self, task_query: str) -> Dict[str, Any]:
        """
        Mark a task as completed.
        
        Args:
            task_query: Query to find the task to complete
            
        Returns:
            Dict with success status and task details
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Find the task
            matching_tasks = await self.search_facts(f"TASK: {task_query}", limit=5)
            
            if not matching_tasks:
                return {
                    "success": False,
                    "message": "No matching task found to complete",
                    "task": None
                }
            
            # Take the most relevant task
            task = matching_tasks[0]
            task_content = task.get("content", "")
            
            # Mark as completed by updating the status
            updated_content = task_content.replace("status: pending", "status: completed")
            updated_content = updated_content.replace("status: in_progress", "status: completed")
            
            # Add completion timestamp
            completion_timestamp = datetime.now().isoformat()
            updated_content += f" [completed: {completion_timestamp}]"
            
            # Store the updated task
            await memory_manager.store_persistent_fact(updated_content, "tasks")
            
            # Store completion record
            completion_record = f"TASK_COMPLETED: {task_query} (completed on {completion_timestamp})"
            await memory_manager.store_persistent_fact(completion_record, "task_completions")
            
            logger.info(f"✅ MemoryAgent completed task: {task_query}")
            
            return {
                "success": True,
                "message": "Task marked as completed",
                "task": task_content,
                "completed_at": completion_timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to complete task: {e}")
            return {
                "success": False,
                "message": f"Error completing task: {str(e)}",
                "task": None
            }
    
    async def update_task(self, task_query: str, **updates) -> Dict[str, Any]:
        """
        Update a task's properties.
        
        Args:
            task_query: Query to find the task to update
            **updates: Properties to update (priority, due_date, project, notes, status)
            
        Returns:
            Dict with success status and updated task details
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Find the task
            matching_tasks = await self.search_facts(f"TASK: {task_query}", limit=5)
            
            if not matching_tasks:
                return {
                    "success": False,
                    "message": "No matching task found to update",
                    "task": None
                }
            
            # Take the most relevant task
            task = matching_tasks[0]
            task_content = task.get("content", "")
            
            # Parse current task data
            task_info = self._parse_task_content(task_content)
            
            # Apply updates
            for key, value in updates.items():
                if key in ["priority", "status", "due_date", "project", "notes"]:
                    task_info[key] = value
            
            # Reconstruct task content
            updated_content = f"TASK: {task_info['task']} (priority: {task_info['priority']}, status: {task_info['status']})"
            if task_info.get("project"):
                updated_content += f" [project: {task_info['project']}]"
            if task_info.get("due_date"):
                updated_content += f" [due: {task_info['due_date']}]"
            if task_info.get("notes"):
                updated_content += f" [notes: {task_info['notes']}]"
            
            # Store the updated task
            await memory_manager.store_persistent_fact(updated_content, "tasks")
            
            # Store update record
            update_record = f"TASK_UPDATED: {task_query} (updated on {datetime.now().isoformat()})"
            await memory_manager.store_persistent_fact(update_record, "task_updates")
            
            logger.info(f"📝 MemoryAgent updated task: {task_query}")
            
            return {
                "success": True,
                "message": "Task updated successfully",
                "old_task": task_content,
                "new_task": updated_content,
                "updated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to update task: {e}")
            return {
                "success": False,
                "message": f"Error updating task: {str(e)}",
                "task": None
            }
    
    async def delete_task(self, task_query: str) -> Dict[str, Any]:
        """
        Delete a task.
        
        Args:
            task_query: Query to find the task to delete
            
        Returns:
            Dict with success status and deleted task details
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Find the task
            matching_tasks = await self.search_facts(f"TASK: {task_query}", limit=5)
            
            if not matching_tasks:
                return {
                    "success": False,
                    "message": "No matching task found to delete",
                    "task": None
                }
            
            # Take the most relevant task
            task = matching_tasks[0]
            task_content = task.get("content", "")
            
            # Store deletion record
            deletion_record = f"TASK_DELETED: {task_content} (deleted on {datetime.now().isoformat()})"
            await memory_manager.store_persistent_fact(deletion_record, "task_deletions")
            
            logger.info(f"🗑️ MemoryAgent deleted task: {task_query}")
            
            return {
                "success": True,
                "message": "Task deleted successfully",
                "deleted_task": task_content,
                "deleted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to delete task: {e}")
            return {
                "success": False,
                "message": f"Error deleting task: {str(e)}",
                "task": None
            }
    
    def _parse_task_content(self, content: str) -> Dict[str, Any]:
        """Parse task content back into structured data"""
        import re
        
        # Content is stored naturally without artificial prefixes
        
        if not content.startswith("TASK:") and not content.startswith("Task:"):
            return None
        
        # Extract task description
        task_match = re.search(r'(?:TASK|Task): (.+?) \(', content)
        if not task_match:
            # Try simpler pattern without parentheses
            task_match = re.search(r'(?:TASK|Task): (.+?)(?:\s*$|\.|\n)', content)
        if not task_match:
            # Try even simpler - everything after TASK:/Task:
            task_match = re.search(r'(?:TASK|Task): (.+)', content)
        task = task_match.group(1).strip() if task_match else ""
        
        # Extract priority
        priority_match = re.search(r'priority: (\w+)', content)
        priority = priority_match.group(1) if priority_match else "medium"
        
        # Extract status
        status_match = re.search(r'status: (\w+)', content)
        status = status_match.group(1) if status_match else "pending"
        
        # Extract project
        project_match = re.search(r'\[project: ([^\]]+)\]', content)
        project = project_match.group(1) if project_match else None
        
        # Extract due date
        due_match = re.search(r'\[due: ([^\]]+)\]', content)
        due_date = due_match.group(1) if due_match else None
        
        # Extract notes
        notes_match = re.search(r'\[notes: ([^\]]+)\]', content)
        notes = notes_match.group(1) if notes_match else None
        
        return {
            "task": task,
            "priority": priority,
            "status": status,
            "project": project,
            "due_date": due_date,
            "notes": notes
        }
    
    # === FILE REFERENCE SYSTEM ===
    
    async def save_file_reference(self, file_path: str, file_type: str = None, purpose: str = None, project: str = None, notes: str = "") -> str:
        """
        Save a file reference for easy retrieval.
        
        Args:
            file_path: Full or relative path to the file
            file_type: Type of file (e.g., "config", "component", "script", "documentation")
            purpose: What the file is used for (e.g., "authentication", "styling", "API")
            project: Project or context the file belongs to
            notes: Additional notes about the file
            
        Returns:
            str: Reference ID if successful, None if failed
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Extract file name and extension
            import os
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1]
            
            # Infer file type from extension if not provided
            if not file_type:
                file_type = self._infer_file_type(file_ext)
            
            # Create structured file reference
            file_ref = f"FILE_REF: {file_path} (type: {file_type})"
            if purpose:
                file_ref += f" [purpose: {purpose}]"
            if project:
                file_ref += f" [project: {project}]"
            if notes:
                file_ref += f" [notes: {notes}]"
            
            # Add timestamp
            file_ref += f" [accessed: {datetime.now().isoformat()}]"
            
            success = await memory_manager.store_persistent_fact(file_ref, "file_references")
            
            if success:
                logger.info(f"📁 MemoryAgent saved file reference: {file_path}")
                return f"file_ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to save file reference: {e}")
            return None
    
    async def find_file(self, query: str, file_type: str = None, project: str = None) -> List[Dict[str, Any]]:
        """
        Find files based on name, purpose, or type.
        
        Args:
            query: Search query (file name, purpose, or general search)
            file_type: Filter by file type (optional)
            project: Filter by project (optional)
            
        Returns:
            List of matching file references
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Build search query
            search_query = f"FILE_REF: {query}"
            if file_type:
                search_query += f" type: {file_type}"
            if project:
                search_query += f" project: {project}"
            
            # Search for file references
            results = await memory_manager.search_memories(search_query, limit=20)
            
            # Parse file reference results
            files = []
            for result in results:
                content = result.get("content", "")
                if content.startswith("FILE_REF:"):
                    file_info = self._parse_file_content(content)
                    if file_info:
                        file_info.update({
                            "id": result.get("uuid", ""),
                            "timestamp": result.get("timestamp", "")
                        })
                        files.append(file_info)
            
            logger.info(f"📁 MemoryAgent found {len(files)} file references for query: {query}")
            return files
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to find files: {e}")
            return []
    
    async def get_recent_files(self, limit: int = 10, file_type: str = None, project: str = None) -> List[Dict[str, Any]]:
        """
        Get recently accessed files.
        
        Args:
            limit: Maximum number of files to return
            file_type: Filter by file type (optional)
            project: Filter by project (optional)
            
        Returns:
            List of recently accessed files
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Build search query for recent files
            search_query = "FILE_REF:"
            if file_type:
                search_query += f" type: {file_type}"
            if project:
                search_query += f" project: {project}"
            
            # Search for file references
            results = await memory_manager.search_memories(search_query, limit=limit * 2)  # Get more to filter
            
            # Parse and sort by access time
            files = []
            for result in results:
                content = result.get("content", "")
                if content.startswith("FILE_REF:"):
                    file_info = self._parse_file_reference(content)
                    if file_info:
                        file_info.update({
                            "id": result.get("uuid", ""),
                            "timestamp": result.get("timestamp", "")
                        })
                        files.append(file_info)
            
            # Sort by access time (most recent first)
            files.sort(key=lambda x: x.get("accessed", ""), reverse=True)
            
            # Return only the requested number
            recent_files = files[:limit]
            
            logger.info(f"📁 MemoryAgent retrieved {len(recent_files)} recent files")
            return recent_files
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to get recent files: {e}")
            return []
    
    async def get_project_files(self, project: str) -> List[Dict[str, Any]]:
        """
        Get all files associated with a specific project.
        
        Args:
            project: Project name
            
        Returns:
            List of files for the project
        """
        return await self.find_file("", project=project)
    
    async def update_file_reference(self, file_path: str, **updates) -> Dict[str, Any]:
        """
        Update a file reference.
        
        Args:
            file_path: Path to the file to update
            **updates: Properties to update (file_type, purpose, project, notes)
            
        Returns:
            Dict with success status and updated reference details
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Find the file reference
            matching_files = await self.find_file(file_path)
            
            if not matching_files:
                return {
                    "success": False,
                    "message": "No matching file reference found to update",
                    "file": None
                }
            
            # Take the most relevant file
            file_ref = matching_files[0]
            old_content = file_ref.get("content", "")
            
            # Parse current file data
            file_info = self._parse_file_reference(old_content)
            
            # Apply updates
            for key, value in updates.items():
                if key in ["file_type", "purpose", "project", "notes"]:
                    file_info[key] = value
            
            # Reconstruct file reference
            updated_content = f"FILE_REF: {file_info['file_path']} (type: {file_info['file_type']})"
            if file_info.get("purpose"):
                updated_content += f" [purpose: {file_info['purpose']}]"
            if file_info.get("project"):
                updated_content += f" [project: {file_info['project']}]"
            if file_info.get("notes"):
                updated_content += f" [notes: {file_info['notes']}]"
            
            # Update access time
            updated_content += f" [accessed: {datetime.now().isoformat()}]"
            
            # Store the updated file reference
            await memory_manager.store_persistent_fact(updated_content, "file_references")
            
            # Store update record
            update_record = f"FILE_UPDATED: {file_path} (updated on {datetime.now().isoformat()})"
            await memory_manager.store_persistent_fact(update_record, "file_updates")
            
            logger.info(f"📝 MemoryAgent updated file reference: {file_path}")
            
            return {
                "success": True,
                "message": "File reference updated successfully",
                "old_reference": old_content,
                "new_reference": updated_content,
                "updated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to update file reference: {e}")
            return {
                "success": False,
                "message": f"Error updating file reference: {str(e)}",
                "file": None
            }
    
    async def delete_file_reference(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a file reference.
        
        Args:
            file_path: Path to the file reference to delete
            
        Returns:
            Dict with success status and deleted reference details
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Find the file reference
            matching_files = await self.find_file(file_path)
            
            if not matching_files:
                return {
                    "success": False,
                    "message": "No matching file reference found to delete",
                    "file": None
                }
            
            # Take the most relevant file
            file_ref = matching_files[0]
            ref_content = file_ref.get("content", "")
            
            # Store deletion record
            deletion_record = f"FILE_DELETED: {ref_content} (deleted on {datetime.now().isoformat()})"
            await memory_manager.store_persistent_fact(deletion_record, "file_deletions")
            
            logger.info(f"🗑️ MemoryAgent deleted file reference: {file_path}")
            
            return {
                "success": True,
                "message": "File reference deleted successfully",
                "deleted_reference": ref_content,
                "deleted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to delete file reference: {e}")
            return {
                "success": False,
                "message": f"Error deleting file reference: {str(e)}",
                "file": None
            }
    
    def _infer_file_type(self, file_ext: str) -> str:
        """Infer file type from extension"""
        ext_mapping = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "react",
            ".ts": "typescript",
            ".tsx": "react_typescript",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".json": "config",
            ".yaml": "config",
            ".yml": "config",
            ".toml": "config",
            ".md": "documentation",
            ".txt": "text",
            ".sql": "database",
            ".sh": "script",
            ".env": "config",
            ".dockerfile": "docker",
            ".gitignore": "config",
            ".README": "documentation"
        }
        
        return ext_mapping.get(file_ext.lower(), "unknown")
    
    def _parse_file_reference(self, content: str) -> Dict[str, Any]:
        """Parse file reference content back into structured data"""
        import re
        
        # Handle FILE_REF: prefix (content is stored naturally)
        if not content.startswith("FILE_REF:"):
            return None
        
        # Extract file path
        path_match = re.search(r'FILE_REF: (.+?) \(', content)
        file_path = path_match.group(1) if path_match else ""
        
        # Extract file type
        type_match = re.search(r'type: (\w+)', content)
        file_type = type_match.group(1) if type_match else "unknown"
        
        # Extract purpose
        purpose_match = re.search(r'\[purpose: ([^\]]+)\]', content)
        purpose = purpose_match.group(1) if purpose_match else None
        
        # Extract project
        project_match = re.search(r'\[project: ([^\]]+)\]', content)
        project = project_match.group(1) if project_match else None
        
        # Extract notes
        notes_match = re.search(r'\[notes: ([^\]]+)\]', content)
        notes = notes_match.group(1) if notes_match else None
        
        # Extract access time
        accessed_match = re.search(r'\[accessed: ([^\]]+)\]', content)
        accessed = accessed_match.group(1) if accessed_match else None
        
        return {
            "file_path": file_path,
            "file_type": file_type,
            "purpose": purpose,
            "project": project,
            "notes": notes,
            "accessed": accessed
        }
    
    async def save_file_reference_with_context(self, file_path: str, file_type: str = None, purpose: str = None, 
                                             project: str = None, semantic_tags: List[str] = None, 
                                             conversation_context: str = "", notes: str = "") -> str:
        """
        🚀 HYBRID STORAGE FIX: Store file references as natural language for semantic search
        
        This fixes the core problem: instead of storing technical formats that Zep can't understand,
        we store natural language descriptions that match how users ask questions.
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Extract file name and extension
            import os
            file_name = os.path.basename(file_path) if not file_path.startswith('http') else file_path
            
            # Infer file type from extension if not provided
            if not file_type and not file_path.startswith('http'):
                file_type = self._infer_file_type(os.path.splitext(file_name)[1])
            elif not file_type:
                file_type = "design" if "figma" in file_path else "other"
            
            # 🚀 HYBRID STORAGE: Create natural language description that semantic search can understand
            if file_type == "design" and "figma" in file_path.lower():
                if purpose:
                    natural_description = f"I have a Figma design file for {purpose}: {file_path}"
                elif project:
                    natural_description = f"I have a Figma design file for the {project} project: {file_path}"
                else:
                    natural_description = f"I have a Figma design file: {file_path}"
            else:
                if purpose:
                    natural_description = f"I have a {file_type} file for {purpose}: {file_path}"
                elif project:
                    natural_description = f"I have a {file_type} file for the {project} project: {file_path}"
                else:
                    natural_description = f"I have a {file_type} file: {file_path}"
            
            # Add semantic tags naturally
            if semantic_tags:
                tag_description = ", ".join(semantic_tags)
                natural_description += f" (tagged: {tag_description})"
            
            # Add notes naturally
            if notes:
                natural_description += f". {notes}"
            
            # Store the natural language description
            success = await memory_manager.store_persistent_fact(natural_description, "files")
            
            # Also store structured metadata for advanced queries (optional)
            if success:
                metadata = {
                    "file_path": file_path,
                    "file_type": file_type,
                    "purpose": purpose,
                    "project": project,
                    "semantic_tags": semantic_tags or [],
                    "accessed": datetime.now().isoformat()
                }
                
                # Store additional semantic facts for different query patterns
                if project:
                    project_fact = f"For the {project} project, I have this file: {file_path}"
                    await memory_manager.store_persistent_fact(project_fact, "project_files")
                
                if purpose:
                    purpose_fact = f"For {purpose}, I have this file: {file_path}"
                    await memory_manager.store_persistent_fact(purpose_fact, "purpose_files")
                
                # Store type-based facts
                type_fact = f"I have a {file_type} file: {file_path}"
                await memory_manager.store_persistent_fact(type_fact, "file_types")
            
            if success:
                logger.info(f"📁 MemoryAgent saved hybrid file reference: {file_path}")
                logger.info(f"   Natural description: {natural_description}")
                return f"file_ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to save hybrid file reference: {e}")
            return None
    
    async def _extract_context_summary(self, conversation_context: str) -> str:
        """Extract key context phrases from conversation for enhanced search"""
        try:
            # Simple keyword extraction for now - could be enhanced with LLM
            import re
            
            # Extract key phrases that might be useful for retrieval
            key_phrases = []
            
            # Look for project/client mentions
            project_matches = re.findall(r'\b(?:for|with|client|project)\s+([A-Z][a-zA-Z\s]+)\b', conversation_context)
            key_phrases.extend(project_matches)
            
            # Look for purpose/function mentions  
            purpose_matches = re.findall(r'\b(?:for|about|regarding)\s+([a-z]+(?:\s+[a-z]+)*)\b', conversation_context.lower())
            key_phrases.extend(purpose_matches)
            
            # Limit and clean
            unique_phrases = list(set(key_phrases))[:3]  # Top 3 unique phrases
            return ', '.join([phrase.strip() for phrase in unique_phrases if len(phrase.strip()) > 2])
            
        except Exception as e:
            logger.warning(f"⚠️ Context extraction failed: {e}")
            return ""
    
    # === RESOURCE/LINK MANAGEMENT ===
    
    async def save_resource(self, url: str, title: str = None, description: str = None, category: str = "general", tags: List[str] = None, notes: str = "") -> str:
        """
        Save a resource/link for later reference.
        
        Args:
            url: The URL to save
            title: Title of the resource (optional)
            description: Description of what the resource is (optional)
            category: Category (e.g., "tool", "reference", "inspiration", "tutorial")
            tags: List of tags for organization (optional)
            notes: Additional notes about why this was saved
            
        Returns:
            str: Resource ID if successful, None if failed
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Extract domain for automatic categorization if not provided
            import re
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            domain = domain_match.group(1) if domain_match else "unknown"
            
            # Infer category from domain if not provided
            if category == "general":
                category = self._infer_resource_category(domain)
            
            # Create structured resource reference
            resource_ref = f"RESOURCE: {url} (category: {category})"
            if title:
                resource_ref += f" [title: {title}]"
            if description:
                resource_ref += f" [description: {description}]"
            if tags:
                resource_ref += f" [tags: {', '.join(tags)}]"
            if notes:
                resource_ref += f" [notes: {notes}]"
            
            # Add domain and timestamp
            resource_ref += f" [domain: {domain}]"
            resource_ref += f" [saved: {datetime.now().isoformat()}]"
            
            success = await memory_manager.store_persistent_fact(resource_ref, "resources")
            
            if success:
                logger.info(f"🔗 MemoryAgent saved resource: {url}")
                return f"resource_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to save resource: {e}")
            return None
    
    async def get_resources(self, category: str = None, tags: List[str] = None, search_query: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get saved resources with optional filtering.
        
        Args:
            category: Filter by category (optional)
            tags: Filter by tags (optional)
            search_query: Search in title, description, or notes (optional)
            limit: Maximum number of resources to return
            
        Returns:
            List of matching resources
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Build search query
            query_parts = ["RESOURCE:"]
            if category:
                query_parts.append(f"category: {category}")
            if tags:
                for tag in tags:
                    query_parts.append(f"tags: {tag}")
            if search_query:
                query_parts.append(search_query)
            
            query = " ".join(query_parts)
            
            # Search for resources
            results = await memory_manager.search_memories(query, limit=limit)
            
            # Parse resource results
            resources = []
            for result in results:
                content = result.get("content", "")
                if content.startswith("RESOURCE:"):
                    resource_info = self._parse_resource_content(content)
                    if resource_info:
                        resource_info.update({
                            "id": result.get("uuid", ""),
                            "timestamp": result.get("timestamp", "")
                        })
                        resources.append(resource_info)
            
            logger.info(f"🔗 MemoryAgent found {len(resources)} resources")
            return resources
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to get resources: {e}")
            return []
    
    async def find_link(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """
        Find links based on URL, title, description, or tags.
        
        Args:
            query: Search query
            category: Filter by category (optional)
            
        Returns:
            List of matching links
        """
        return await self.get_resources(category=category, search_query=query)
    
    async def get_resources_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all resources in a specific category.
        
        Args:
            category: Category to search for
            
        Returns:
            List of resources in the category
        """
        return await self.get_resources(category=category)
    
    async def get_resources_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """
        Get resources that match any of the given tags.
        
        Args:
            tags: List of tags to search for
            
        Returns:
            List of resources matching the tags
        """
        return await self.get_resources(tags=tags)
    
    async def update_resource(self, url: str, **updates) -> Dict[str, Any]:
        """
        Update a resource's metadata.
        
        Args:
            url: URL of the resource to update
            **updates: Properties to update (title, description, category, tags, notes)
            
        Returns:
            Dict with success status and updated resource details
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Find the resource
            matching_resources = await self.find_link(url)
            
            if not matching_resources:
                return {
                    "success": False,
                    "message": "No matching resource found to update",
                    "resource": None
                }
            
            # Take the most relevant resource
            resource = matching_resources[0]
            old_content = resource.get("content", "")
            
            # Parse current resource data
            resource_info = self._parse_resource_content(old_content)
            
            # Apply updates
            for key, value in updates.items():
                if key in ["title", "description", "category", "tags", "notes"]:
                    resource_info[key] = value
            
            # Reconstruct resource reference
            updated_content = f"RESOURCE: {resource_info['url']} (category: {resource_info['category']})"
            if resource_info.get("title"):
                updated_content += f" [title: {resource_info['title']}]"
            if resource_info.get("description"):
                updated_content += f" [description: {resource_info['description']}]"
            if resource_info.get("tags"):
                if isinstance(resource_info["tags"], list):
                    updated_content += f" [tags: {', '.join(resource_info['tags'])}]"
                else:
                    updated_content += f" [tags: {resource_info['tags']}]"
            if resource_info.get("notes"):
                updated_content += f" [notes: {resource_info['notes']}]"
            
            # Preserve domain and update timestamp
            if resource_info.get("domain"):
                updated_content += f" [domain: {resource_info['domain']}]"
            updated_content += f" [saved: {datetime.now().isoformat()}]"
            
            # Store the updated resource
            await memory_manager.store_persistent_fact(updated_content, "resources")
            
            # Store update record
            update_record = f"RESOURCE_UPDATED: {url} (updated on {datetime.now().isoformat()})"
            await memory_manager.store_persistent_fact(update_record, "resource_updates")
            
            logger.info(f"📝 MemoryAgent updated resource: {url}")
            
            return {
                "success": True,
                "message": "Resource updated successfully",
                "old_resource": old_content,
                "new_resource": updated_content,
                "updated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to update resource: {e}")
            return {
                "success": False,
                "message": f"Error updating resource: {str(e)}",
                "resource": None
            }
    
    async def delete_resource(self, url: str) -> Dict[str, Any]:
        """
        Delete a saved resource.
        
        Args:
            url: URL of the resource to delete
            
        Returns:
            Dict with success status and deleted resource details
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Find the resource
            matching_resources = await self.find_link(url)
            
            if not matching_resources:
                return {
                    "success": False,
                    "message": "No matching resource found to delete",
                    "resource": None
                }
            
            # Take the most relevant resource
            resource = matching_resources[0]
            resource_content = resource.get("content", "")
            
            # Store deletion record
            deletion_record = f"RESOURCE_DELETED: {resource_content} (deleted on {datetime.now().isoformat()})"
            await memory_manager.store_persistent_fact(deletion_record, "resource_deletions")
            
            logger.info(f"🗑️ MemoryAgent deleted resource: {url}")
            
            return {
                "success": True,
                "message": "Resource deleted successfully",
                "deleted_resource": resource_content,
                "deleted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to delete resource: {e}")
            return {
                "success": False,
                "message": f"Error deleting resource: {str(e)}",
                "resource": None
            }
    
    def _infer_resource_category(self, domain: str) -> str:
        """Infer resource category from domain"""
        domain_mapping = {
            "github.com": "tool",
            "stackoverflow.com": "reference",
            "docs.python.org": "documentation",
            "developer.mozilla.org": "documentation",
            "react.dev": "documentation",
            "nextjs.org": "documentation",
            "tailwindcss.com": "documentation",
            "youtube.com": "tutorial",
            "medium.com": "article",
            "dev.to": "article",
            "hackernews.com": "news",
            "reddit.com": "discussion",
            "figma.com": "design",
            "dribbble.com": "inspiration",
            "behance.net": "inspiration",
            "codepen.io": "tool",
            "codesandbox.io": "tool",
            "vercel.com": "tool",
            "netlify.com": "tool",
            "heroku.com": "tool",
            "aws.amazon.com": "tool",
            "cloud.google.com": "tool"
        }
        
        return domain_mapping.get(domain.lower(), "general")
    
    def _parse_resource_content(self, content: str) -> Dict[str, Any]:
        """Parse resource content back into structured data"""
        import re
        
        # Handle RESOURCE: prefix (content is stored naturally)
        
        if not content.startswith("RESOURCE:"):
            return None
        
        # Extract URL
        url_match = re.search(r'RESOURCE: (.+?) \(', content)
        url = url_match.group(1) if url_match else ""
        
        # Extract category
        category_match = re.search(r'category: (\w+)', content)
        category = category_match.group(1) if category_match else "general"
        
        # Extract title
        title_match = re.search(r'\[title: ([^\]]+)\]', content)
        title = title_match.group(1) if title_match else None
        
        # Extract description
        desc_match = re.search(r'\[description: ([^\]]+)\]', content)
        description = desc_match.group(1) if desc_match else None
        
        # Extract tags
        tags_match = re.search(r'\[tags: ([^\]]+)\]', content)
        tags = tags_match.group(1).split(', ') if tags_match else []
        
        # Extract notes
        notes_match = re.search(r'\[notes: ([^\]]+)\]', content)
        notes = notes_match.group(1) if notes_match else None
        
        # Extract domain
        domain_match = re.search(r'\[domain: ([^\]]+)\]', content)
        domain = domain_match.group(1) if domain_match else None
        
        # Extract saved time
        saved_match = re.search(r'\[saved: ([^\]]+)\]', content)
        saved = saved_match.group(1) if saved_match else None
        
        return {
            "url": url,
            "category": category,
            "title": title,
            "description": description,
            "tags": tags,
            "notes": notes,
            "domain": domain,
            "saved": saved
        }

    async def save_resource_with_context(self, url: str, title: str = None, description: str = None, 
                                       category: str = "general", semantic_tags: List[str] = None, 
                                       conversation_context: str = "", notes: str = "") -> str:
        """
        🚀 ENHANCED RESOURCE STORAGE: Save resource with semantic context and intelligent tags
        
        This replaces the basic save_resource with contextual understanding for better retrieval.
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Extract domain for automatic categorization if not provided
            import re
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            domain = domain_match.group(1) if domain_match else "unknown"
            
            # Infer category from domain if not provided or is general
            if category == "general":
                category = self._infer_resource_category(domain)
            
            # Create enhanced resource reference with semantic information
            resource_ref = f"RESOURCE: {url} (category: {category})"
            if title:
                resource_ref += f" [title: {title}]"
            if description:
                resource_ref += f" [description: {description}]"
            if semantic_tags:
                resource_ref += f" [tags: {', '.join(semantic_tags)}]"
            if notes:
                resource_ref += f" [notes: {notes}]"
            
            # Add contextual information for better retrieval
            if conversation_context:
                # Extract key phrases from conversation for search enhancement
                context_summary = await self._extract_context_summary(conversation_context)
                if context_summary:
                    resource_ref += f" [context: {context_summary}]"
            
            # Add domain and timestamp
            resource_ref += f" [domain: {domain}]"
            resource_ref += f" [saved: {datetime.now().isoformat()}]"
            
            # Store the enhanced resource
            success = await memory_manager.store_persistent_fact(resource_ref, "resources")
            
            # Also store semantic facts for different retrieval paths
            if success and semantic_tags:
                for tag in semantic_tags:
                    semantic_fact = f"Resource tagged '{tag}': {url}"
                    if title:
                        semantic_fact += f" ({title})"
                    await memory_manager.store_persistent_fact(semantic_fact, "semantic_tags")
            
            # Store category-based fact for category filtering
            if success and category != "general":
                category_fact = f"Resource in {category} category: {url}"
                if title:
                    category_fact += f" ({title})"
                await memory_manager.store_persistent_fact(category_fact, "resource_categories")
            
            # Store description-based fact for purpose-based retrieval
            if success and description:
                purpose_fact = f"Resource for {description}: {url}"
                if title:
                    purpose_fact += f" ({title})"
                await memory_manager.store_persistent_fact(purpose_fact, "resource_purposes")
            
            if success:
                logger.info(f"🔗 MemoryAgent saved enhanced resource: {url}")
                logger.info(f"   Category: {category}, Title: {title}, Tags: {semantic_tags}")
                return f"resource_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to save enhanced resource: {e}")
            return None

    # === GENERAL MEMORY OPERATIONS ===
    
    async def search(self, query: str, scope: str = "all", limit: int = 10) -> List[Dict[str, Any]]:
        """
        General memory search across different scopes.
        
        Args:
            query: Search query
            scope: Search scope ("facts", "preferences", "time", "all")
            limit: Maximum results
            
        Returns:
            List of matching entries
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # Adjust query based on scope
            if scope == "facts":
                search_query = f"FACT: {query}"
            elif scope == "preferences":
                search_query = f"PREFERENCE: {query}"
            elif scope == "time":
                search_query = f"time {query}"
            else:
                search_query = query
            
            results = await memory_manager.search_memories(search_query, limit, None)
            logger.info(f"🔍 MemoryAgent searched {scope}: {len(results)} results for '{query}'")
            return results
        except Exception as e:
            logger.error(f"❌ MemoryAgent search failed: {e}")
            return []
    
    async def get_context(self, session_id: str, query: str = "") -> str:
        """
        Get relevant context for a conversation session.
        
        Args:
            session_id: Current conversation session
            query: Optional query to focus the context
            
        Returns:
            Formatted context string
        """
        try:
            memory_manager = await self._get_memory_manager()
            context = await memory_manager.get_memory_context(session_id, query)
            logger.info(f"📝 MemoryAgent retrieved context for session: {session_id}")
            return context
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to get context: {e}")
            return ""

    async def save_task_with_context(self, task: str, priority: str = "medium", due_date: str = None, 
                                   project: str = None, semantic_tags: List[str] = None, 
                                   conversation_context: str = "", notes: str = "") -> str:
        """
        🚀 HYBRID STORAGE FIX: Store tasks as natural language for semantic search
        
        This fixes the core problem: instead of storing artificial formats,
        we store natural language descriptions that match how users ask questions.
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # 🚀 HYBRID STORAGE: Create natural language description that semantic search can understand
            if priority == "urgent":
                natural_description = f"Urgent task: {task}"
            elif priority == "high":
                natural_description = f"High priority task: {task}"
            elif priority == "low":
                natural_description = f"Low priority task: {task}"
            else:
                natural_description = f"Task: {task}"
            
            # Add due date naturally
            if due_date:
                natural_description += f" (due {due_date})"
            
            # Add project context naturally
            if project:
                natural_description += f" for the {project} project"
            
            # Add notes naturally
            if notes:
                natural_description += f". {notes}"
            
            # Add semantic tags naturally
            if semantic_tags:
                tag_description = ", ".join(semantic_tags)
                natural_description += f" (tagged: {tag_description})"
            
            # Store the natural language description
            success = await memory_manager.store_persistent_fact(natural_description, "tasks")
            
            # Also store structured metadata for advanced queries (optional)
            if success:
                # Store additional semantic facts for different query patterns
                if project:
                    project_fact = f"For the {project} project, I need to: {task}"
                    await memory_manager.store_persistent_fact(project_fact, "project_tasks")
                
                if priority in ["urgent", "high"]:
                    priority_fact = f"{priority.capitalize()} priority: {task}"
                    await memory_manager.store_persistent_fact(priority_fact, "priority_tasks")
                
                # Store general task reference
                task_fact = f"To do: {task}"
                await memory_manager.store_persistent_fact(task_fact, "todo_items")
            
            if success:
                logger.info(f"✅ MemoryAgent saved hybrid task: {task}")
                logger.info(f"   Natural description: {natural_description}")
                return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ MemoryAgent failed to save hybrid task: {e}")
            return None

    async def find_files_by_intent(self, query: str, context: str = "") -> List[Dict[str, Any]]:
        """
        🚀 HYBRID SEARCH FIX: Find files using natural language queries that match storage format
        
        This fixes the core problem: search queries that match the natural language storage format.
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # 🚀 INTELLIGENT MODEL SELECTION: Use LLM to understand search intent
            from agents.model_selector import model_selector
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            from config import settings
            
            selected_model = model_selector.select_model(
                user_input=query,
                agent_name="MemoryAgent",
                context=f"file intent search: {context}"
            )
            
            llm = ChatOpenAI(
                model=selected_model,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1  # Low temperature for predictable search terms
            )
            
            search_prompt = f"""You are a semantic file search system. Generate search queries that match the EXACT storage format.

USER QUERY: "{query}"
CONTEXT: "{context}"

FILES ARE NOW STORED IN NATURAL LANGUAGE FORMAT:
- "I have a Figma design file for homepage: https://www.figma.com/design/homepage"
- "I have a design file for the homepage project: https://www.figma.com/design/homepage"
- "I have a Figma design file: https://www.figma.com/design/homepage (tagged: Figma, design, homepage, UI)"
- "For the homepage project, I have this file: https://www.figma.com/design/homepage"
- "I have a design file: https://www.figma.com/design/homepage"

CRITICAL: Generate search queries that will find files by matching the natural language storage:

For "design files" queries, include:
1. "I have a design file" - matches storage format
2. "I have a Figma design file" - for Figma files specifically
3. "design file" - general term
4. "Figma design" - for Figma files
5. "design" - broad match

For "homepage" queries, include:
1. "homepage" - direct match
2. "I have a Figma design file for homepage" - exact match
3. "homepage project" - project match
4. "for homepage" - purpose match
5. "tagged: homepage" - tag match

EXAMPLES:
- "homepage file" → ["homepage", "I have a Figma design file for homepage", "homepage project", "for homepage", "tagged: homepage"]
- "design files" → ["I have a design file", "I have a Figma design file", "design file", "Figma design", "design"]
- "what design files do i have" → ["I have a design file", "I have a Figma design file", "design file", "Figma design", "design"]

Return ONLY a JSON array of search queries:
["query1", "query2", "query3", "query4", "query5"]
"""

            messages = [HumanMessage(content=search_prompt)]
            response = await llm.ainvoke(messages)
            
            # Parse search queries
            import json
            try:
                search_queries = json.loads(response.content)
                if not isinstance(search_queries, list):
                    search_queries = [query]  # Fallback to original query
            except json.JSONDecodeError:
                search_queries = [query]  # Fallback to original query
            
            # Execute multiple searches and combine results
            all_results = []
            seen_paths = set()
            
            for search_query in search_queries:
                try:
                    results = await memory_manager.search_memories(search_query, limit=10)
                    for result in results:
                        content = result.get("content", "")
                        
                        # Check for natural language file descriptions
                        file_info = None
                        if ("I have a" in content and "file" in content) or ("For the" in content and "file" in content):
                            file_info = self._parse_natural_file_content(content)
                        
                        if file_info and file_info["file_path"] not in seen_paths:
                            file_info.update({
                                "id": result.get("uuid", ""),
                                "timestamp": result.get("timestamp", ""),
                                "relevance_score": 1.0  # Could be enhanced with semantic scoring
                            })
                            all_results.append(file_info)
                            seen_paths.add(file_info["file_path"])
                except Exception as e:
                    logger.warning(f"⚠️ Search query failed: {search_query} - {e}")
            
            # Sort by relevance (could be enhanced with semantic scoring)
            all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            logger.info(f"✅ Found {len(all_results)} total files for intent: '{query}'")
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Find files by intent failed: {e}")
            return []
    
    async def find_tasks_by_intent(self, query: str, context: str = "") -> List[Dict[str, Any]]:
        """
        🚀 HYBRID SEARCH FIX: Find tasks using natural language queries that match storage format
        
        This fixes the core problem: search queries that match the natural language storage format.
        """
        try:
            memory_manager = await self._get_memory_manager()
            
            # 🚀 INTELLIGENT MODEL SELECTION: Use LLM to understand search intent
            from agents.model_selector import model_selector
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            from config import settings
            
            selected_model = model_selector.select_model(
                user_input=query,
                agent_name="MemoryAgent",
                context=f"task intent search: {context}"
            )
            
            llm = ChatOpenAI(
                model=selected_model,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1
            )
            
            search_prompt = f"""You are a semantic task search system. Generate search queries that match the EXACT storage format.

USER QUERY: "{query}"
CONTEXT: "{context}"

TASKS ARE NOW STORED IN NATURAL LANGUAGE FORMAT:
- "Task: Fix the authentication bug"
- "Urgent task: Fix the authentication bug"
- "High priority task: Fix the authentication bug"
- "Task: Fix the authentication bug (due tomorrow) for the homepage project"
- "For the homepage project, I need to: Fix the authentication bug"
- "To do: Fix the authentication bug"

CRITICAL: Generate search queries that will find tasks by matching the natural language storage:

For general task queries, include:
1. "Task:" - matches storage format
2. "Urgent task:" - for urgent tasks
3. "High priority task:" - for high priority tasks
4. "To do:" - alternative format
5. "task" - broad match

For priority-based queries, include:
1. "Urgent task:" - exact match
2. "High priority task:" - exact match
3. "urgent" - general term
4. "high priority" - general term
5. "priority" - broad match

For project-based queries, include:
1. "for the [project] project" - exact match
2. "[project] project" - project match
3. "For the [project] project, I need to:" - alternative format

EXAMPLES:
- "what tasks do I have" → ["Task:", "Urgent task:", "High priority task:", "To do:", "task"]
- "urgent tasks" → ["Urgent task:", "urgent", "High priority task:", "priority", "Task:"]
- "homepage tasks" → ["homepage project", "for the homepage project", "Task:", "To do:", "homepage"]

Return ONLY a JSON array of search queries:
["query1", "query2", "query3", "query4", "query5"]
"""
            
            messages = [HumanMessage(content=search_prompt)]
            response = await llm.ainvoke(messages)
            
            # Parse search queries
            import json
            try:
                search_queries = json.loads(response.content)
                if not isinstance(search_queries, list):
                    search_queries = [query]  # Fallback to original query
            except json.JSONDecodeError:
                search_queries = [query]  # Fallback to original query
            
            # Execute multiple searches and combine results
            all_results = []
            seen_tasks = set()
            
            for search_query in search_queries:
                try:
                    results = await memory_manager.search_memories(search_query, limit=10)
                    for result in results:
                        content = result.get("content", "")
                        
                        # Check for natural language task descriptions
                        task_info = None
                        if ("Task:" in content or "task:" in content or 
                            "Urgent task:" in content or "High priority task:" in content or
                            "To do:" in content or "For the" in content and "I need to:" in content):
                            task_info = self._parse_natural_task_content(content)
                        
                        if task_info and task_info["task"] not in seen_tasks:
                            task_info.update({
                                "id": result.get("uuid", ""),
                                "timestamp": result.get("timestamp", ""),
                                "relevance_score": 1.0
                            })
                            all_results.append(task_info)
                            seen_tasks.add(task_info["task"])
                except Exception as e:
                    logger.warning(f"⚠️ Task search for query '{search_query}' failed: {e}")

            # Sort by priority and relevance
            priority_order = {"urgent": 4, "high": 3, "medium": 2, "low": 1}
            all_results.sort(key=lambda x: (
                priority_order.get(x.get("priority", "medium"), 2),
                x.get("relevance_score", 0)
            ), reverse=True)

            logger.info(f"✅ Found {len(all_results)} total tasks for intent: '{query}'")
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Find tasks by intent failed: {e}")
            return []

    async def find_resources_by_intent(self, query: str, context: str = "") -> List[Dict[str, Any]]:
        """
        🚀 SEMANTIC RESOURCE SEARCH: Find resources using natural language intent.
        
        This method uses an LLM to generate multiple intelligent search queries 
        and then searches memory for any of them.
        """
        all_results = []
        seen_urls = set()
        
        try:
            # Lazy import to avoid circular dependencies
            from agents.model_selector import model_selector
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            from config import settings
            memory_manager = await self._get_memory_manager()

            # 🚀 INTELLIGENT MODEL SELECTION
            selected_model = model_selector.select_model(
                user_input=query,
                agent_name="MemoryAgent", 
                context=f"resource intent search: {context}"
            )
            
            # Create LLM with selected model
            llm = ChatOpenAI(
                model=selected_model,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1  # Low temperature for predictable search terms
            )
            
            search_prompt = f"""You are a semantic resource search system. Understand what the user is looking for and generate the best search queries.

USER QUERY: "{query}"
CONTEXT: "{context}"

Generate 3-5 different search queries that could find relevant resources (links, tools, articles). Think about:
1. **Direct Format Match**: ALWAYS include a query in the format 'RESOURCE: [key_term]' or 'http'.
2. **Domain/Site Search**: Search for the website domain (e.g., "figma.com", "grafana.com").
3. **Purpose/Category**: Search for what the resource is for (e.g., "design tool", "monitoring", "tutorial").
4. **Title/Content Search**: Use keywords from the likely title or description.
5. **Tag-based Searches**: Use semantic tags.

Examples:
- "grafana link" → ["RESOURCE: grafana", "grafana.com", "monitoring tool", "dashboards"]
- "design tools" → ["RESOURCE: design", "category: design", "figma.com", "sketch.com", "UI/UX tools"]
- "show me react tutorials" → ["RESOURCE: react", "react.dev", "tutorial", "javascript framework", "tags: tutorial"]

Return ONLY a JSON array of search queries:
["query1", "query2", "query3", "query4", "query5"]
"""
            
            logger.info(f"🔍 GENERATING RESOURCE SEARCH QUERIES: '{query[:100]}...'")
            messages = [HumanMessage(content=search_prompt)]
            response = await llm.ainvoke(messages)
            
            import json
            search_queries = json.loads(response.content)
            
            logger.info(f"🧠 Generated Resource Search Queries: {search_queries}")

            for search_query in search_queries:
                try:
                    results = await memory_manager.search_memories(search_query, limit=10)
                    for result in results:
                        content = result.get("content", "")
                        # This should be adapted to parse natural language resources if the format changes
                        if content.startswith("RESOURCE:"):
                            resource_info = self._parse_resource_content(content)
                            if resource_info and resource_info["url"] not in seen_urls:
                                resource_info.update({
                                    "id": result.get("uuid", ""),
                                    "timestamp": result.get("timestamp", ""),
                                    "relevance_score": 1.0
                                })
                                all_results.append(resource_info)
                                seen_urls.add(resource_info["url"])
                except Exception as e:
                    logger.warning(f"⚠️ Resource search for query '{search_query}' failed: {e}")
            
            logger.info(f"✅ Found {len(all_results)} total resources for intent: '{query}'")
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Find resources by intent failed: {e}")
            return []

    def _parse_file_content(self, content: str) -> Dict[str, Any]:
        """Parse file reference content"""
        import re
        
        # Handle FILE_REF: prefix (content is stored naturally)
        if not content.startswith("FILE_REF:"):
            return None
        
        # Extract file path
        path_match = re.search(r'FILE_REF: (.+?) \(', content)
        file_path = path_match.group(1) if path_match else ""
        
        # Extract all fields including semantic tags
        file_type = self._extract_field(content, "type")
        purpose = self._extract_field(content, "purpose")
        project = self._extract_field(content, "project")
        tags = self._extract_field(content, "tags")
        context = self._extract_field(content, "context")
        accessed = self._extract_field(content, "accessed")
        
        # Parse tags as list
        if tags:
            tags = [tag.strip() for tag in tags.split(",")]
        else:
            tags = []
        
        return {
            "file_path": file_path,
            "file_type": file_type or "other",
            "purpose": purpose,
            "project": project,
            "semantic_tags": tags,
            "context": context,
            "accessed": accessed
        }
    
    def _parse_semantic_file_fact(self, content: str) -> Dict[str, Any]:
        """Parse semantic file facts to extract file information."""
        try:
            # Handle different semantic file fact formats:
            # "FACT: File tagged 'homepage': https://www.figma.com/design/test-focused-storage-retrieval for homepage redesign"
            # "FACT: File for homepage redesign: https://www.figma.com/design/test-focused-storage-retrieval in homepage"
            # "FACT: File in homepage category: https://www.figma.com/design/test-focused-storage-retrieval"
            
            # Content is stored naturally without artificial prefixes
            content = content.strip()
            
            # Extract file path (look for URLs or file paths)
            import re
            
            # Look for URLs first
            url_match = re.search(r'https?://[^\s]+', content)
            if url_match:
                file_path = url_match.group(0)
            else:
                # Look for file paths
                file_match = re.search(r'[^\s:]+\.[a-zA-Z0-9]+', content)
                if file_match:
                    file_path = file_match.group(0)
                else:
                    return {}
            
            # Extract purpose/description
            purpose = ""
            if "for " in content:
                purpose_match = re.search(r'for ([^(]+)', content)
                if purpose_match:
                    purpose = purpose_match.group(1).strip()
            
            # Extract project
            project = ""
            if "in " in content:
                project_match = re.search(r'in ([^(]+)', content)
                if project_match:
                    project = project_match.group(1).strip()
            
            # Extract tags from content
            tags = []
            if "tagged" in content:
                tag_match = re.search(r"tagged ['\"]([^'\"]+)['\"]", content)
                if tag_match:
                    tags.append(tag_match.group(1))
            
            # Infer file type
            file_type = "other"
            if "figma.com" in file_path.lower():
                file_type = "design"
            elif "github.com" in file_path.lower():
                file_type = "code"
            elif file_path.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                file_type = "code"
            elif file_path.endswith(('.md', '.txt', '.doc', '.docx')):
                file_type = "documentation"
            
            return {
                "file_path": file_path,
                "file_type": file_type,
                "purpose": purpose,
                "project": project,
                "semantic_tags": tags,
                "context": content,
                "accessed": None
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse semantic file fact: {content} - {e}")
            return {}
    
    def _extract_field(self, content: str, field_name: str) -> str:
        """Helper to extract a field value from bracketed content."""
        import re
        match = re.search(rf'\[{field_name}: (.*?)\]', content)
        return match.group(1).strip() if match else ""

    def _parse_natural_task_content(self, content: str) -> Dict[str, Any]:
        """
        🚀 NATURAL TASK PARSER: Parse natural language task descriptions
        
        This replaces the old LLM-based parser with a simple regex-based one for the new format.
        """
        import re
        
        # Extract task description
        task = ""
        priority = "medium"
        
        # Check for priority-based task formats
        if "Urgent task:" in content:
            priority = "urgent"
            task_match = re.search(r'Urgent task:\s*(.+?)(?:\s*\(|$)', content)
            task = task_match.group(1).strip() if task_match else ""
        elif "High priority task:" in content:
            priority = "high"
            task_match = re.search(r'High priority task:\s*(.+?)(?:\s*\(|$)', content)
            task = task_match.group(1).strip() if task_match else ""
        elif "Low priority task:" in content:
            priority = "low"
            task_match = re.search(r'Low priority task:\s*(.+?)(?:\s*\(|$)', content)
            task = task_match.group(1).strip() if task_match else ""
        elif "Task:" in content:
            task_match = re.search(r'Task:\s*(.+?)(?:\s*\(|$)', content)
            task = task_match.group(1).strip() if task_match else ""
        elif "To do:" in content:
            task_match = re.search(r'To do:\s*(.+?)(?:\s*\(|$)', content)
            task = task_match.group(1).strip() if task_match else ""
        elif "For the" in content and "I need to:" in content:
            task_match = re.search(r'I need to:\s*(.+?)(?:\s*\(|$)', content)
            task = task_match.group(1).strip() if task_match else ""
        
        # Extract due date (look for "due [date]")
        due_match = re.search(r'due\s+([^)]+)', content)
        due_date = due_match.group(1).strip() if due_match else None
        
        # Extract project (look for "for the [project] project")
        project_match = re.search(r'for the (\w+) project', content)
        project = project_match.group(1) if project_match else None
        
        # Extract tags (look for "tagged:" mentions)
        tags_match = re.search(r'tagged:\s*([^)]+)', content)
        tags = []
        if tags_match:
            tags = [tag.strip() for tag in tags_match.group(1).split(',')]
        
        # Extract notes (everything after the main task description)
        notes_match = re.search(r'\.\s*(.+?)(?:\s*\(tagged:|$)', content)
        notes = notes_match.group(1).strip() if notes_match else ""
        
        return {
            "task": task,
            "priority": priority,
            "due_date": due_date,
            "project": project,
            "semantic_tags": tags,
            "notes": notes
        }

    def _parse_natural_file_content(self, content: str) -> Dict[str, Any]:
        """
        🚀 ENHANCED PARSING: Parse natural language file descriptions from memory storage
        
        Handles formats like:
        - "Resource tagged 'navigation': https://www.figma.com/design/..."
        - "Resource in design category: https://www.figma.com/design/..."
        - "Resource for A Figma design file: https://www.figma.com/design/..."
        - "I have a Figma design file for homepage: https://www.figma.com/design/..."
        """
        import re
        
        try:
            # 🚨 CRITICAL FIX: Handle all the natural language formats we found in Zep
            
            # Pattern 1: "Resource tagged 'X': URL (Description)"
            pattern1 = r"Resource tagged ['\"]([^'\"]+)['\"]:\s*(https?://[^\s]+)\s*\(([^)]+)\)"
            match1 = re.search(pattern1, content)
            if match1:
                tag, url, description = match1.groups()
                return {
                    "file_path": url.strip(),
                    "purpose": description.strip(),
                    "semantic_tags": [tag.strip()],
                    "file_type": "design" if "figma" in url.lower() else "other",
                    "context": f"Tagged as {tag}",
                    "project": self._extract_project_from_description(description)
                }
            
            # Pattern 2: "Resource in X category: URL (Description)"
            pattern2 = r"Resource in ([^:]+) category:\s*(https?://[^\s]+)\s*\(([^)]+)\)"
            match2 = re.search(pattern2, content)
            if match2:
                category, url, description = match2.groups()
                return {
                    "file_path": url.strip(),
                    "purpose": description.strip(),
                    "semantic_tags": [category.strip()],
                    "file_type": category.strip() if category.strip() in ["design", "config", "component"] else "other",
                    "context": f"Categorized as {category}",
                    "project": self._extract_project_from_description(description)
                }
            
            # Pattern 3: "Resource for X: URL (Description)"
            pattern3 = r"Resource for ([^:]+):\s*(https?://[^\s]+)\s*\(([^)]+)\)"
            match3 = re.search(pattern3, content)
            if match3:
                purpose, url, description = match3.groups()
                return {
                    "file_path": url.strip(),
                    "purpose": purpose.strip(),
                    "semantic_tags": self._extract_tags_from_purpose(purpose),
                    "file_type": "design" if "figma" in url.lower() or "design" in purpose.lower() else "other",
                    "context": f"Purpose: {purpose}",
                    "project": self._extract_project_from_description(description)
                }
            
            # Pattern 4: Generic URL extraction with context
            url_pattern = r"(https?://[^\s\)]+)"
            url_match = re.search(url_pattern, content)
            if url_match:
                url = url_match.group(1)
                
                # Extract context around the URL
                context_before = content[:url_match.start()].strip()
                context_after = content[url_match.end():].strip()
                
                # Determine file type and purpose from context
                file_type = "other"
                purpose = "unknown"
                tags = []
                
                if "figma" in url.lower():
                    file_type = "design"
                    purpose = "design file"
                    tags = ["Figma", "design"]
                
                if "design" in content.lower():
                    file_type = "design"
                    tags.append("design")
                
                if "homepage" in content.lower():
                    purpose = "homepage design"
                    tags.append("homepage")
                
                return {
                    "file_path": url.strip(),
                    "purpose": purpose,
                    "semantic_tags": tags,
                    "file_type": file_type,
                    "context": content.strip()[:100] + ("..." if len(content) > 100 else ""),
                    "project": self._extract_project_from_content(content)
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse natural file content: {e}")
            return None

    def _extract_project_from_description(self, description: str) -> str:
        """Extract project from description"""
        import re
        match = re.search(r'for the (\w+) project', description)
        return match.group(1) if match else "unknown"

    def _extract_project_from_content(self, content: str) -> str:
        """Extract project from content"""
        import re
        match = re.search(r'for the (\w+) project', content)
        return match.group(1) if match else "unknown"

    def _extract_tags_from_purpose(self, purpose: str) -> List[str]:
        """Extract tags from purpose"""
        import re
        match = re.findall(r'\b(?:for|about|regarding)\s+([a-z]+(?:\s+[a-z]+)*)\b', purpose)
        return [tag.strip() for tag in match]

# Global instance
memory_agent = MemoryAgent() 