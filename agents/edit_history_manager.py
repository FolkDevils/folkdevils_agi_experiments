from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
import os
import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class EditHistoryEntry:
    """Represents a single edit operation in the history"""
    edit_id: str
    session_id: str
    timestamp: datetime
    instruction: str
    instruction_type: str  # replace, remove, shorten, rewrite
    before_content: str
    after_content: str
    before_version: int
    after_version: int
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EditHistoryEntry':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass 
class EditHistoryState:
    """Complete edit history for a session"""
    session_id: str
    entries: List[EditHistoryEntry]
    total_edits: int
    first_edit: Optional[datetime] = None
    last_edit: Optional[datetime] = None
    
    def add_entry(self, entry: EditHistoryEntry):
        """Add a new edit entry to the history"""
        self.entries.append(entry)
        self.total_edits = len(self.entries)
        
        if not self.first_edit or entry.timestamp < self.first_edit:
            self.first_edit = entry.timestamp
        if not self.last_edit or entry.timestamp > self.last_edit:
            self.last_edit = entry.timestamp
    
    def get_last_successful_edit(self) -> Optional[EditHistoryEntry]:
        """Get the most recent successful edit (excluding undos)"""
        for entry in reversed(self.entries):
            if entry.success and entry.instruction_type != "undo":
                return entry
        return None
    
    def get_version_content(self, version: int) -> Optional[str]:
        """Get the content at a specific version"""
        for entry in self.entries:
            if entry.after_version == version and entry.success:
                return entry.after_content
        return None
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "total_edits": self.total_edits,
            "first_edit": self.first_edit.isoformat() if self.first_edit else None,
            "last_edit": self.last_edit.isoformat() if self.last_edit else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EditHistoryState':
        entries = [EditHistoryEntry.from_dict(entry_data) for entry_data in data.get('entries', [])]
        first_edit = datetime.fromisoformat(data['first_edit']) if data.get('first_edit') else None
        last_edit = datetime.fromisoformat(data['last_edit']) if data.get('last_edit') else None
        
        return cls(
            session_id=data['session_id'],
            entries=entries,
            total_edits=data.get('total_edits', len(entries)),
            first_edit=first_edit,
            last_edit=last_edit
        )

class EditHistoryManager:
    """
    Manages edit history for precision editing sessions.
    Tracks every edit operation with full before/after states.
    """
    
    def __init__(self, storage_dir: str = "memory_storage"):
        self.storage_dir = storage_dir
        self.history_cache: Dict[str, EditHistoryState] = {}
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"EditHistoryManager initialized with storage: {storage_dir}")
    
    def _get_history_file_path(self, session_id: str) -> str:
        """Get the file path for storing edit history"""
        return os.path.join(self.storage_dir, f"edit_history_{session_id}.json")
    
    async def load_edit_history(self, session_id: str) -> EditHistoryState:
        """Load edit history for a session"""
        if session_id in self.history_cache:
            return self.history_cache[session_id]
        
        file_path = self._get_history_file_path(session_id)
        
        try:
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    history = EditHistoryState.from_dict(data)
                    self.history_cache[session_id] = history
                    logger.info(f"Loaded edit history for session {session_id}: {len(history.entries)} entries")
                    return history
            else:
                # Create new history
                history = EditHistoryState(session_id=session_id, entries=[], total_edits=0)
                self.history_cache[session_id] = history
                logger.info(f"Created new edit history for session {session_id}")
                return history
                
        except Exception as e:
            logger.error(f"Failed to load edit history for {session_id}: {e}")
            # Return empty history on error
            history = EditHistoryState(session_id=session_id, entries=[], total_edits=0)
            self.history_cache[session_id] = history
            return history
    
    async def save_edit_history(self, history: EditHistoryState) -> bool:
        """Save edit history to disk"""
        try:
            file_path = self._get_history_file_path(history.session_id)
            
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(history.to_dict(), indent=2))
            
            logger.info(f"Saved edit history for session {history.session_id}: {len(history.entries)} entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save edit history for {history.session_id}: {e}")
            return False
    
    async def record_edit(self, 
                         session_id: str,
                         instruction: str,
                         instruction_type: str,
                         before_content: str,
                         after_content: str,
                         before_version: int,
                         after_version: int,
                         success: bool = True,
                         error_message: Optional[str] = None) -> EditHistoryEntry:
        """Record a new edit operation in the history"""
        
        # Load current history
        history = await self.load_edit_history(session_id)
        
        # Create new edit entry
        edit_id = f"{session_id}_edit_{len(history.entries) + 1}_{int(datetime.now().timestamp())}"
        
        entry = EditHistoryEntry(
            edit_id=edit_id,
            session_id=session_id,
            timestamp=datetime.now(),
            instruction=instruction,
            instruction_type=instruction_type,
            before_content=before_content,
            after_content=after_content,
            before_version=before_version,
            after_version=after_version,
            success=success,
            error_message=error_message
        )
        
        # Add to history
        history.add_entry(entry)
        
        # Update cache
        self.history_cache[session_id] = history
        
        # Save to disk
        await self.save_edit_history(history)
        
        logger.info(f"Recorded edit: {instruction_type} operation for session {session_id}")
        return entry
    
    async def get_edit_history(self, session_id: str, limit: Optional[int] = None) -> List[EditHistoryEntry]:
        """Get edit history for a session"""
        history = await self.load_edit_history(session_id)
        
        if limit:
            return history.entries[-limit:]
        return history.entries
    
    async def get_last_edit(self, session_id: str) -> Optional[EditHistoryEntry]:
        """Get the most recent edit for a session"""
        history = await self.load_edit_history(session_id)
        return history.get_last_successful_edit()
    
    async def undo_last_edit(self, session_id: str) -> Optional[str]:
        """Undo the last edit and return the previous content"""
        history = await self.load_edit_history(session_id)
        
        if not history.entries:
            return None
        
        # Find the last successful edit that hasn't been undone
        # Count how many undos we've done
        undo_count = sum(1 for entry in history.entries if entry.instruction_type == "undo" and entry.success)
        
        # Get all successful non-undo edits
        edits = [entry for entry in history.entries if entry.success and entry.instruction_type != "undo"]
        
        # If we've already undone all edits, return None
        if undo_count >= len(edits):
            return None
        
        # Get the edit to undo (counting backwards from the most recent)
        edit_to_undo = edits[-(undo_count + 1)]
        
        # Return the before content of that edit
        return edit_to_undo.before_content
    
    async def get_version_content(self, session_id: str, version: int) -> Optional[str]:
        """Get content at a specific version"""
        history = await self.load_edit_history(session_id)
        return history.get_version_content(version)
    
    async def get_edit_statistics(self, session_id: str) -> Dict:
        """Get statistics about edits for a session"""
        history = await self.load_edit_history(session_id)
        
        if not history.entries:
            return {
                "total_edits": 0,
                "successful_edits": 0,
                "failed_edits": 0,
                "edit_types": {},
                "first_edit": None,
                "last_edit": None
            }
        
        successful_edits = sum(1 for entry in history.entries if entry.success)
        failed_edits = len(history.entries) - successful_edits
        
        edit_types = {}
        for entry in history.entries:
            edit_types[entry.instruction_type] = edit_types.get(entry.instruction_type, 0) + 1
        
        return {
            "total_edits": history.total_edits,
            "successful_edits": successful_edits,
            "failed_edits": failed_edits,
            "edit_types": edit_types,
            "first_edit": history.first_edit.isoformat() if history.first_edit else None,
            "last_edit": history.last_edit.isoformat() if history.last_edit else None
        }

    async def clear_history(self, session_id: str) -> bool:
        """Clear edit history for a session (for testing)"""
        try:
            # Remove from cache
            if session_id in self.history_cache:
                del self.history_cache[session_id]
            
            # Remove file if it exists
            file_path = self._get_history_file_path(session_id)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logger.info(f"Cleared edit history for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear edit history for {session_id}: {e}")
            return False

# Global instance
edit_history_manager = EditHistoryManager() 