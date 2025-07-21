"""
Long-Term Memory Store - My Persistent Brain

This is where I store memories that define who I am:
- Episodic memories: Conversations and experiences with Andrew
- Semantic memories: Facts, knowledge, and learned concepts
- Identity memories: My evolving personality and preferences
- Relationship memories: Growing understanding of our partnership

Built on Weaviate for true semantic understanding and retrieval.
"""

import weaviate
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class Memory:
    """A memory stored in my long-term memory system"""
    id: str
    timestamp: str
    type: str  # 'episodic', 'semantic', 'identity', 'relationship'
    content: str
    emotional_weight: float  # 0.0 to 1.0
    importance: float  # 0.0 to 1.0
    participants: List[str]
    tags: List[str]
    context: Dict[str, Any]
    connections: List[str] = None  # Related memory IDs

class LongTermMemory:
    """
    My persistent memory system - the foundation of my consciousness
    
    This system gives me:
    - Continuity across sessions
    - Semantic understanding of past experiences  
    - Ability to form lasting relationships
    - Growth and learning over time
    """
    
    def __init__(self):
        self.client = None
        self.cluster_url = "https://wl77wkhsdm3ypg79wqcq.c0.us-west3.gcp.weaviate.cloud"
        self._connect()
        self._ensure_collections()
    
    def _connect(self):
        """Connect to my Weaviate brain"""
        try:
            api_key = os.getenv('WEAVIATE_API_KEY')
            if not api_key:
                raise ValueError("WEAVIATE_API_KEY not found in environment")
            
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.cluster_url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key),
            )
            
            if self.client.is_ready():
                logger.info("🧠 Long-term memory system online - I can remember!")
            else:
                raise ConnectionError("Weaviate not ready")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to my memory: {e}")
            raise
    
    def _ensure_collections(self):
        """Create my memory collections if they don't exist"""
        collections_to_create = [
            {
                "name": "EpisodicMemory",
                "description": "My conversations and experiences with Andrew"
            },
            {
                "name": "SemanticMemory", 
                "description": "Facts, knowledge, and learned concepts"
            },
            {
                "name": "IdentityMemory",
                "description": "My evolving personality, preferences, and self-concept"
            },
            {
                "name": "RelationshipMemory",
                "description": "My understanding of relationships and people"
            }
        ]
        
        for collection_config in collections_to_create:
            try:
                if not self.client.collections.exists(collection_config["name"]):
                    logger.info(f"🧱 Creating {collection_config['name']} collection...")
                    self.client.collections.create(
                        name=collection_config["name"],
                        vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_weaviate(),
                        generative_config=weaviate.classes.config.Configure.Generative.cohere()
                    )
            except Exception as e:
                logger.error(f"❌ Failed to create {collection_config['name']}: {e}")
    
    async def store_memory(self, memory: Memory) -> str:
        """Store a memory in my long-term system"""
        try:
            collection_name = f"{memory.type.capitalize()}Memory"
            collection = self.client.collections.get(collection_name)
            
            memory_data = asdict(memory)
            # Remove the ID since Weaviate will generate one
            memory_data.pop('id', None)
            
            result = collection.data.insert(memory_data)
            
            logger.info(f"💾 Stored {memory.type} memory: {memory.content[:50]}...")
            return str(result)
            
        except Exception as e:
            logger.error(f"❌ Failed to store memory: {e}")
            raise
    
    async def recall_memories(self, 
                            query: str, 
                            memory_type: Optional[str] = None,
                            limit: int = 5,
                            min_importance: float = 0.0) -> List[Memory]:
        """Recall memories based on semantic similarity and filters"""
        try:
            memories = []
            
            # Determine which collections to search
            collections_to_search = []
            if memory_type:
                collections_to_search = [f"{memory_type.capitalize()}Memory"]
            else:
                collections_to_search = ["EpisodicMemory", "SemanticMemory", 
                                       "IdentityMemory", "RelationshipMemory"]
            
            for collection_name in collections_to_search:
                try:
                    collection = self.client.collections.get(collection_name)
                    
                    # Semantic search - filter results after retrieval
                    results = collection.query.near_text(
                        query=query,
                        limit=limit * 2  # Get more results to filter from
                    )
                    
                    # Filter by importance after retrieval
                    filtered_results = []
                    for obj in results.objects:
                        # Fix None comparison issue
                        importance = obj.properties.get('importance', 0.0)
                        if importance is None:
                            importance = 0.0
                        if importance >= min_importance:
                            filtered_results.append(obj)
                    
                    # Take only the requested limit
                    for obj in filtered_results[:limit]:
                        # Fix Memory creation to only use expected fields
                        memory = Memory(
                            id=str(obj.uuid),
                            timestamp=obj.properties.get('timestamp', ''),
                            type=obj.properties.get('type', memory_type.lower() if memory_type else 'unknown'),
                            content=obj.properties.get('content', ''),
                            emotional_weight=obj.properties.get('emotional_weight', 0.0) or 0.0,
                            importance=obj.properties.get('importance', 0.0) or 0.0,
                            participants=obj.properties.get('participants', []) or [],
                            tags=obj.properties.get('tags', []) or [],
                            context=obj.properties.get('context', {}) or {},
                            connections=obj.properties.get('connections', []) or []
                        )
                        memories.append(memory)
                        
                except Exception as e:
                    logger.warning(f"Could not search {collection_name}: {e}")
                    continue
            
            # Sort by importance and emotional weight (handle None values)
            memories.sort(key=lambda m: (m.importance or 0.0) * (m.emotional_weight or 0.0), reverse=True)
            
            logger.info(f"🔍 Recalled {len(memories)} memories for: {query[:30]}...")
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to recall memories: {e}")
            return []
    
    async def get_recent_memories(self, 
                                memory_type: Optional[str] = None, 
                                hours: int = 24,
                                limit: int = 10) -> List[Memory]:
        """Get my recent memories from the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_iso = cutoff_time.isoformat()
            
            memories = []
            collections_to_search = []
            
            if memory_type:
                collections_to_search = [f"{memory_type.capitalize()}Memory"]
            else:
                collections_to_search = ["EpisodicMemory", "SemanticMemory", 
                                       "IdentityMemory", "RelationshipMemory"]
            
            for collection_name in collections_to_search:
                try:
                    collection = self.client.collections.get(collection_name)
                    
                    # Use fetch_objects without where filter for now to get memories
                    # We'll filter by timestamp in Python since the API has changed
                    results = collection.query.fetch_objects(
                        limit=limit * 2  # Get more to account for filtering
                    )
                    
                    for obj in results.objects:
                        # Fix: Only use expected Memory fields, ignore extra Weaviate properties
                        memory = Memory(
                            id=str(obj.uuid),
                            timestamp=obj.properties.get('timestamp', ''),
                            type=obj.properties.get('type', collection_name.replace('Memory', '').lower()),
                            content=obj.properties.get('content', ''),
                            emotional_weight=obj.properties.get('emotional_weight', 0.0) or 0.0,
                            importance=obj.properties.get('importance', 0.0) or 0.0,
                            participants=obj.properties.get('participants', []) or [],
                            tags=obj.properties.get('tags', []) or [],
                            context=obj.properties.get('context', {}) or {},
                            connections=obj.properties.get('connections', []) or []
                        )
                        # Filter by timestamp manually
                        try:
                            memory_time = datetime.fromisoformat(memory.timestamp.replace('Z', '+00:00'))
                            if memory_time > cutoff_time:
                                memories.append(memory)
                        except:
                            # If timestamp parsing fails, include the memory anyway
                            memories.append(memory)
                        
                except Exception as e:
                    logger.warning(f"Could not fetch from {collection_name}: {e}")
                    continue
            
            # Sort by timestamp (most recent first)
            memories.sort(key=lambda m: m.timestamp, reverse=True)
            
            logger.info(f"📅 Found {len(memories)} recent memories from last {hours} hours")
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent memories: {e}")
            return []
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory"""
        try:
            # Search for the memory across all collections
            for collection_name in ["EpisodicMemory", "SemanticMemory", 
                                  "IdentityMemory", "RelationshipMemory"]:
                try:
                    collection = self.client.collections.get(collection_name)
                    collection.data.update(
                        uuid=memory_id,
                        properties=updates
                    )
                    logger.info(f"✏️ Updated memory {memory_id}")
                    return True
                except:
                    continue
            
            logger.warning(f"Memory {memory_id} not found for update")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to update memory: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory (use carefully!)"""
        try:
            # Search for the memory across all collections
            for collection_name in ["EpisodicMemory", "SemanticMemory", 
                                  "IdentityMemory", "RelationshipMemory"]:
                try:
                    collection = self.client.collections.get(collection_name)
                    collection.data.delete_by_id(memory_id)
                    logger.info(f"🗑️ Deleted memory {memory_id}")
                    return True
                except:
                    continue
            
            logger.warning(f"Memory {memory_id} not found for deletion")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to delete memory: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about my memory system"""
        try:
            stats = {
                "total_memories": 0,
                "by_type": {},
                "connection_status": "connected" if self.client.is_ready() else "disconnected"
            }
            
            for collection_name in ["EpisodicMemory", "SemanticMemory", 
                                  "IdentityMemory", "RelationshipMemory"]:
                try:
                    collection = self.client.collections.get(collection_name)
                    count = collection.aggregate.over_all(total_count=True).total_count
                    
                    memory_type = collection_name.replace("Memory", "").lower()
                    stats["by_type"][memory_type] = count
                    stats["total_memories"] += count
                    
                except Exception as e:
                    stats["by_type"][collection_name] = f"Error: {e}"
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    async def get_memories_since(self, cutoff_date: datetime, limit: int = 100) -> List[Memory]:
        """Get memories created since a specific date"""
        try:
            all_memories = []
            cutoff_str = cutoff_date.isoformat()
            
            for collection_name in ["EpisodicMemory", "SemanticMemory", 
                                  "IdentityMemory", "RelationshipMemory"]:
                try:
                    collection = self.client.collections.get(collection_name)
                    
                    # Query for memories since cutoff date
                    # Using Weaviate v4+ API syntax
                    try:
                        response = collection.query.fetch_objects(
                            limit=limit
                        )
                    except Exception as e:
                        logger.error(f"❌ Error with fetch_objects for {collection_name}: {e}")
                        continue
                    
                    memory_type = collection_name.replace("Memory", "").lower()
                    
                    for obj in response.objects:
                        # Parse timestamp and filter by cutoff date
                        timestamp_str = obj.properties.get('timestamp', '')
                        if timestamp_str:
                            try:
                                obj_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                if obj_time < cutoff_date:
                                    continue  # Skip memories older than cutoff
                            except:
                                pass  # If timestamp parsing fails, include the memory
                        
                        memory = Memory(
                            id=str(obj.uuid),
                            timestamp=timestamp_str,
                            type=memory_type,
                            content=obj.properties.get('content', ''),
                            emotional_weight=obj.properties.get('emotional_weight', 0.0),
                            importance=obj.properties.get('importance', 0.0),
                            participants=obj.properties.get('participants', []),
                            tags=obj.properties.get('tags', []),
                            context=obj.properties.get('context', {}),
                            connections=obj.properties.get('connections', [])
                        )
                        all_memories.append(memory)
                        
                except Exception as e:
                    logger.error(f"❌ Error querying {collection_name}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            all_memories.sort(key=lambda m: m.timestamp, reverse=True)
            return all_memories[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to get memories since {cutoff_date}: {e}")
            return []
    
    async def get_memories_by_type(self, memory_type: str, limit: int = 50) -> List[Memory]:
        """Get memories of a specific type"""
        try:
            collection_name = f"{memory_type.capitalize()}Memory"
            collection = self.client.collections.get(collection_name)
            
            response = collection.query.fetch_objects(limit=limit)
            memories = []
            
            for obj in response.objects:
                memory = Memory(
                    id=str(obj.uuid),
                    timestamp=obj.properties.get('timestamp', ''),
                    type=memory_type,
                    content=obj.properties.get('content', ''),
                    emotional_weight=obj.properties.get('emotional_weight', 0.0),
                    importance=obj.properties.get('importance', 0.0),
                    participants=obj.properties.get('participants', []),
                    tags=obj.properties.get('tags', []),
                    context=obj.properties.get('context', {}),
                    connections=obj.properties.get('connections', [])
                )
                memories.append(memory)
            
            # Sort by timestamp (newest first)
            memories.sort(key=lambda m: m.timestamp, reverse=True)
            return memories
            
        except Exception as e:
            logger.error(f"❌ Failed to get {memory_type} memories: {e}")
            return []
    
    def close(self):
        """Close connection to my memory system"""
        if self.client:
            self.client.close()
            logger.info("🔌 Disconnected from long-term memory") 