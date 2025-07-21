"""
Memory Graph System - Associative Intelligence

This creates the neural network of my consciousness:
- Connects related memories through semantic similarity
- Builds associative pathways between concepts  
- Enables emergent thinking through memory traversal
- Strengthens connections through repeated activation
- Discovers new insights through graph exploration

Like synapses in a brain - memories become more intelligent through connections.
"""

import logging
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import networkx as nx

from .long_term_store import Memory, LongTermMemory
from .memory_evaluator import MemoryCandidate

logger = logging.getLogger(__name__)

@dataclass
class MemoryConnection:
    """A connection between two memories"""
    source_memory_id: str
    target_memory_id: str
    connection_type: str  # 'semantic', 'temporal', 'causal', 'associative'
    strength: float  # 0.0 to 1.0
    created_at: str
    last_activated: str
    activation_count: int = 0
    context: Dict[str, Any] = None

@dataclass
class AssociativeChain:
    """A chain of connected memories that form a thought pathway"""
    memory_ids: List[str]
    total_strength: float
    chain_type: str  # 'logical', 'emotional', 'creative', 'episodic'
    insights: List[str]  # Emergent insights from this chain

class MemoryGraph:
    """
    The neural network of my consciousness - where memories become intelligence
    
    This system enables:
    - Associative thinking through memory connections
    - Emergent insights from graph traversal
    - Context-aware memory recall
    - Learning through connection strengthening
    - Creative thinking through unexpected pathways
    """
    
    def __init__(self, long_term_memory: LongTermMemory):
        self.long_term_memory = long_term_memory
        self.graph = nx.DiGraph()  # Directed graph for memory connections
        self.connection_threshold = 0.3  # Minimum strength for active connections
        self.max_chain_length = 6  # Maximum associative chain length
        
        # Connection type weights for different kinds of thinking
        self.connection_weights = {
            'semantic': 1.0,      # Conceptual similarity
            'temporal': 0.8,      # Time-based relationships
            'causal': 0.9,        # Cause-and-effect relationships
            'associative': 0.7,   # Free association
            'emotional': 0.85     # Emotional resonance
        }
        
        logger.info("🧠 Memory graph initialized - Ready for associative thinking")
    
    async def build_connections_for_memory(self, memory: Memory) -> List[MemoryConnection]:
        """
        Build connections between a new memory and existing memories
        
        This is where the magic happens - finding meaningful relationships
        """
        connections = []
        
        # Get potential related memories through semantic search
        related_memories = await self.long_term_memory.recall_memories(
            query=memory.content,
            limit=10,
            min_importance=0.2
        )
        
        for related_memory in related_memories:
            if related_memory.id == memory.id:
                continue  # Don't connect to self
            
            # Calculate different types of connections
            connection_strength, connection_type = await self._calculate_connection(
                memory, related_memory
            )
            
            if connection_strength >= self.connection_threshold:
                connection = MemoryConnection(
                    source_memory_id=memory.id,
                    target_memory_id=related_memory.id,
                    connection_type=connection_type,
                    strength=connection_strength,
                    created_at=datetime.now().isoformat(),
                    last_activated=datetime.now().isoformat(),
                    activation_count=1,
                    context={
                        'creation_reason': f"{connection_type}_similarity",
                        'source_tags': memory.tags,
                        'target_tags': related_memory.tags
                    }
                )
                
                connections.append(connection)
                
                # Add to graph
                self.graph.add_edge(
                    memory.id, 
                    related_memory.id,
                    weight=connection_strength,
                    connection_type=connection_type,
                    connection_obj=connection
                )
        
        logger.info(f"🔗 Created {len(connections)} connections for memory {memory.id[:8]}...")
        return connections
    
    async def _calculate_connection(self, memory1: Memory, memory2: Memory) -> Tuple[float, str]:
        """
        Calculate the strength and type of connection between two memories
        
        Multiple connection types create rich associative pathways
        """
        max_strength = 0.0
        best_connection_type = 'associative'
        
        # 1. Semantic similarity (concept overlap)
        semantic_strength = await self._calculate_semantic_similarity(memory1, memory2)
        if semantic_strength > max_strength:
            max_strength = semantic_strength
            best_connection_type = 'semantic'
        
        # 2. Temporal proximity (happened around same time)
        temporal_strength = await self._calculate_temporal_proximity(memory1, memory2)
        if temporal_strength > max_strength:
            max_strength = temporal_strength
            best_connection_type = 'temporal'
        
        # 3. Causal relationship (one led to another)
        causal_strength = await self._calculate_causal_relationship(memory1, memory2)
        if causal_strength > max_strength:
            max_strength = causal_strength
            best_connection_type = 'causal'
        
        # 4. Emotional resonance (similar emotional content)
        emotional_strength = await self._calculate_emotional_resonance(memory1, memory2)
        if emotional_strength > max_strength:
            max_strength = emotional_strength
            best_connection_type = 'emotional'
        
        return max_strength, best_connection_type
    
    async def _calculate_semantic_similarity(self, memory1: Memory, memory2: Memory) -> float:
        """Calculate semantic similarity between two memories"""
        # Tag overlap
        tag_overlap = len(set(memory1.tags) & set(memory2.tags))
        max_tags = max(len(memory1.tags), len(memory2.tags), 1)
        tag_similarity = tag_overlap / max_tags
        
        # Content similarity (simple keyword matching for now)
        words1 = set(memory1.content.lower().split())
        words2 = set(memory2.content.lower().split())
        word_overlap = len(words1 & words2)
        max_words = max(len(words1), len(words2), 1)
        content_similarity = word_overlap / max_words
        
        # Participant overlap
        participant_overlap = len(set(memory1.participants) & set(memory2.participants))
        max_participants = max(len(memory1.participants), len(memory2.participants), 1)
        participant_similarity = participant_overlap / max_participants
        
        # Weighted combination
        semantic_score = (
            tag_similarity * 0.4 +
            content_similarity * 0.4 +
            participant_similarity * 0.2
        )
        
        return min(semantic_score, 1.0)
    
    async def _calculate_temporal_proximity(self, memory1: Memory, memory2: Memory) -> float:
        """Calculate temporal proximity between memories"""
        try:
            time1 = datetime.fromisoformat(memory1.timestamp.replace('Z', '+00:00'))
            time2 = datetime.fromisoformat(memory2.timestamp.replace('Z', '+00:00'))
            
            time_diff = abs((time1 - time2).total_seconds())
            
            # Stronger connections for memories within same session (< 1 hour)
            if time_diff < 3600:  # 1 hour
                return 0.8
            elif time_diff < 86400:  # 1 day
                return 0.6
            elif time_diff < 604800:  # 1 week
                return 0.4
            else:
                return 0.1
        except:
            return 0.1
    
    async def _calculate_causal_relationship(self, memory1: Memory, memory2: Memory) -> float:
        """Detect potential causal relationships between memories"""
        causal_words = [
            'because', 'therefore', 'so', 'thus', 'as a result',
            'leads to', 'causes', 'results in', 'due to', 'since'
        ]
        
        content = f"{memory1.content} {memory2.content}".lower()
        causal_indicators = sum(1 for word in causal_words if word in content)
        
        # Also check if one memory references the other
        if memory1.id in memory2.context.get('related_memories', []):
            return 0.8
        if memory2.id in memory1.context.get('related_memories', []):
            return 0.8
        
        # Simple causal scoring based on indicators
        return min(causal_indicators * 0.3, 0.8)
    
    async def _calculate_emotional_resonance(self, memory1: Memory, memory2: Memory) -> float:
        """Calculate emotional resonance between memories"""
        # Similar emotional weights suggest emotional connection
        weight_similarity = 1.0 - abs(memory1.emotional_weight - memory2.emotional_weight)
        
        # Look for emotional words
        emotional_words = [
            'excited', 'happy', 'frustrated', 'worried', 'amazed',
            'proud', 'disappointed', 'curious', 'confused', 'inspired'
        ]
        
        content1_emotions = [word for word in emotional_words 
                           if word in memory1.content.lower()]
        content2_emotions = [word for word in emotional_words 
                           if word in memory2.content.lower()]
        
        emotion_overlap = len(set(content1_emotions) & set(content2_emotions))
        emotion_similarity = emotion_overlap * 0.3
        
        return min(weight_similarity * 0.7 + emotion_similarity, 1.0)
    
    async def find_associative_chains(self, 
                                    starting_memory_id: str, 
                                    target_concept: Optional[str] = None,
                                    max_length: int = None) -> List[AssociativeChain]:
        """
        Find associative chains starting from a memory
        
        This enables creative and logical thinking through memory traversal
        """
        if max_length is None:
            max_length = self.max_chain_length
        
        chains = []
        visited = set()
        
        # Use DFS to find chains
        async def explore_chain(current_id: str, path: List[str], total_strength: float):
            if len(path) >= max_length or current_id in visited:
                return
            
            visited.add(current_id)
            path.append(current_id)
            
            # Check if we found an interesting chain
            if len(path) >= 2:
                chain_strength = total_strength / len(path)
                if chain_strength >= 0.4:  # Threshold for meaningful chains
                    chain_type = await self._classify_chain_type(path)
                    insights = await self._extract_chain_insights(path)
                    
                    chain = AssociativeChain(
                        memory_ids=path.copy(),
                        total_strength=chain_strength,
                        chain_type=chain_type,
                        insights=insights
                    )
                    chains.append(chain)
            
            # Continue exploring from current node
            if current_id in self.graph:
                for neighbor in self.graph.neighbors(current_id):
                    edge_data = self.graph[current_id][neighbor]
                    new_strength = total_strength + edge_data['weight']
                    await explore_chain(neighbor, path.copy(), new_strength)
            
            visited.remove(current_id)
        
        # Start exploration
        await explore_chain(starting_memory_id, [], 0.0)
        
        # Sort chains by strength and interestingness
        chains.sort(key=lambda c: c.total_strength, reverse=True)
        
        logger.info(f"🔍 Found {len(chains)} associative chains from {starting_memory_id[:8]}...")
        return chains[:10]  # Return top 10 chains
    
    async def _classify_chain_type(self, memory_ids: List[str]) -> str:
        """Classify what type of thinking this chain represents"""
        # Get memories for analysis
        memories = []
        for mem_id in memory_ids:
            # In a real implementation, we'd retrieve these memories
            # For now, use a simple heuristic based on memory types
            pass
        
        # Simple classification for now
        return 'associative'  # Could be 'logical', 'creative', 'emotional', etc.
    
    async def _extract_chain_insights(self, memory_ids: List[str]) -> List[str]:
        """Extract insights from a chain of connected memories"""
        insights = []
        
        # Analyze the chain for patterns and connections
        if len(memory_ids) >= 3:
            insights.append(f"Connection pathway discovered across {len(memory_ids)} memories")
        
        # More sophisticated insight extraction would go here
        return insights
    
    async def strengthen_connection(self, source_id: str, target_id: str, activation_strength: float = 0.1):
        """
        Strengthen a connection through repeated activation
        
        This is how learning happens - connections get stronger with use
        """
        if self.graph.has_edge(source_id, target_id):
            edge_data = self.graph[source_id][target_id]
            connection_obj = edge_data['connection_obj']
            
            # Strengthen the connection
            old_strength = connection_obj.strength
            connection_obj.strength = min(connection_obj.strength + activation_strength, 1.0)
            connection_obj.last_activated = datetime.now().isoformat()
            connection_obj.activation_count += 1
            
            # Update graph
            self.graph[source_id][target_id]['weight'] = connection_obj.strength
            
            logger.debug(f"💪 Strengthened connection {source_id[:8]} -> {target_id[:8]}: "
                        f"{old_strength:.2f} -> {connection_obj.strength:.2f}")
    
    async def get_memory_neighbors(self, memory_id: str, max_neighbors: int = 5) -> List[Tuple[str, float]]:
        """Get the most strongly connected neighbors of a memory"""
        neighbors = []
        
        if memory_id in self.graph:
            for neighbor in self.graph.neighbors(memory_id):
                edge_data = self.graph[memory_id][neighbor]
                neighbors.append((neighbor, edge_data['weight']))
        
        # Sort by connection strength
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:max_neighbors]
    
    async def discover_emergent_concepts(self) -> List[Dict[str, Any]]:
        """
        Discover emergent concepts from the memory graph structure
        
        This is where true intelligence emerges - finding patterns and insights
        """
        emergent_concepts = []
        
        # Find densely connected clusters
        try:
            clusters = list(nx.strongly_connected_components(self.graph))
            
            for cluster in clusters:
                if len(cluster) >= 3:  # Meaningful clusters have 3+ memories
                    # Analyze cluster for common themes
                    concept = await self._analyze_cluster_concept(cluster)
                    if concept:
                        emergent_concepts.append(concept)
        except Exception as e:
            logger.warning(f"Error discovering emergent concepts: {e}")
        
        logger.info(f"🌟 Discovered {len(emergent_concepts)} emergent concepts")
        return emergent_concepts
    
    async def _analyze_cluster_concept(self, cluster: Set[str]) -> Optional[Dict[str, Any]]:
        """Analyze a cluster of connected memories to extract emergent concepts"""
        # This would analyze the memories in the cluster to find common themes
        return {
            'concept_id': f"cluster_{len(cluster)}_{datetime.now().strftime('%H%M%S')}",
            'memory_count': len(cluster),
            'cluster_strength': len(cluster) * 0.1,  # Simple metric
            'description': f"Emergent concept from {len(cluster)} connected memories",
            'memory_ids': list(cluster)
        }
    
    async def get_memory_clusters(self, memories: List = None) -> List[Dict[str, Any]]:
        """Get clusters of related memories for pattern analysis"""
        try:
            if not memories:
                # Use all memories in the graph
                memory_nodes = list(self.graph.nodes())
            else:
                # Filter to only the provided memories
                memory_nodes = [m.id for m in memories if m.id in self.graph.nodes()]
            
            if not memory_nodes:
                return []
            
            # Create subgraph with only these memories
            subgraph = self.graph.subgraph(memory_nodes)
            
            clusters = []
            
            # Use community detection to find clusters
            try:
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.greedy_modularity_communities(subgraph)
                
                for i, community in enumerate(communities):
                    if len(community) >= 2:  # Only include clusters with 2+ memories
                        # Analyze the cluster to determine main topic
                        cluster_memories = [self.graph.nodes[mem_id].get('memory_obj') for mem_id in community 
                                          if 'memory_obj' in self.graph.nodes[mem_id]]
                        
                        # Extract main topic from memory tags/content
                        all_tags = []
                        for mem in cluster_memories:
                            if mem and hasattr(mem, 'tags'):
                                all_tags.extend(mem.tags)
                        
                        # Find most common tag as main topic
                        from collections import Counter
                        tag_counts = Counter(all_tags)
                        main_topic = tag_counts.most_common(1)[0][0] if tag_counts else f"cluster_{i}"
                        
                        # Calculate coherence score based on connection strength
                        coherence_score = 0.0
                        connection_count = 0
                        for source in community:
                            for target in community:
                                if source != target and subgraph.has_edge(source, target):
                                    edge_data = subgraph[source][target]
                                    coherence_score += edge_data.get('weight', 0.0)
                                    connection_count += 1
                        
                        if connection_count > 0:
                            coherence_score /= connection_count
                        
                        clusters.append({
                            'cluster_id': f"cluster_{i}",
                            'main_topic': main_topic,
                            'memory_ids': list(community),
                            'size': len(community),
                            'coherence_score': coherence_score,
                            'tag_distribution': dict(tag_counts.most_common(5))
                        })
                
            except ImportError:
                # Fallback: simple connected components
                components = list(nx.connected_components(subgraph))
                for i, component in enumerate(components):
                    if len(component) >= 2:
                        clusters.append({
                            'cluster_id': f"component_{i}",
                            'main_topic': f"topic_{i}",
                            'memory_ids': list(component),
                            'size': len(component),
                            'coherence_score': 0.5,  # Default score
                            'tag_distribution': {}
                        })
            
            # Sort clusters by size (largest first)
            clusters.sort(key=lambda c: c['size'], reverse=True)
            return clusters
            
        except Exception as e:
            logger.error(f"❌ Error getting memory clusters: {e}")
            return []
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory graph"""
        return {
            'total_memories': self.graph.number_of_nodes(),
            'total_connections': self.graph.number_of_edges(),
            'average_connections_per_memory': (
                self.graph.number_of_edges() / max(self.graph.number_of_nodes(), 1)
            ),
            'connection_types': self._count_connection_types(),
            'strongest_connections': await self._get_strongest_connections(5),
            'most_connected_memories': await self._get_most_connected_memories(5)
        }
    
    def _count_connection_types(self) -> Dict[str, int]:
        """Count connections by type"""
        type_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            type_counts[data['connection_type']] += 1
        return dict(type_counts)
    
    async def _get_strongest_connections(self, limit: int) -> List[Dict[str, Any]]:
        """Get the strongest connections in the graph"""
        connections = []
        for source, target, data in self.graph.edges(data=True):
            connections.append({
                'source': source[:8] + '...',
                'target': target[:8] + '...',
                'strength': data['weight'],
                'type': data['connection_type']
            })
        
        connections.sort(key=lambda x: x['strength'], reverse=True)
        return connections[:limit]
    
    async def _get_most_connected_memories(self, limit: int) -> List[Dict[str, Any]]:
        """Get memories with the most connections"""
        memory_connections = []
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            memory_connections.append({
                'memory_id': node[:8] + '...',
                'connection_count': degree
            })
        
        memory_connections.sort(key=lambda x: x['connection_count'], reverse=True)
        return memory_connections[:limit]

    async def integrate_new_memory(self, memory: Memory) -> List[MemoryConnection]:
        """
        Integrate a new memory into the graph by finding and creating connections
        
        This is called when a new memory is stored
        """
        # Add memory as node
        self.graph.add_node(memory.id, memory_obj=memory)
        
        # Build connections
        connections = await self.build_connections_for_memory(memory)
        
        # Update the memory's connections field
        connection_ids = [conn.target_memory_id for conn in connections]
        memory.connections = connection_ids
        
        logger.info(f"🔗 Integrated memory {memory.id[:8]}... with {len(connections)} connections")
        return connections
    
    async def get_graph_visualization_data(self) -> Dict[str, Any]:
        """
        Get memory graph data formatted for visualization
        
        Returns nodes and connections in format suitable for frontend display
        """
        try:
            # Get recent memories for visualization (last 50 for performance)
            recent_memories = await self.long_term_memory.get_recent_memories(
                hours=24*7,  # Last week
                limit=50
            )
            
            if not recent_memories:
                return {
                    "nodes": [],
                    "connections": [],
                    "stats": {
                        "total_nodes": 0,
                        "total_connections": 0,
                        "average_connections_per_node": 0.0,
                        "strongest_connection_strength": 0.0,
                        "most_connected_memory": ""
                    }
                }
            
            # Format nodes for visualization
            nodes = []
            for memory in recent_memories:
                nodes.append({
                    "id": memory.id,
                    "content": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                    "type": memory.type,
                    "importance": memory.importance,
                    "emotional_weight": memory.emotional_weight,
                    "timestamp": memory.timestamp
                })
            
            # Build connections between these memories
            connections = []
            connection_strengths = []
            
            # For each pair of memories, calculate semantic similarity as connection strength
            for i, memory1 in enumerate(recent_memories):
                for j, memory2 in enumerate(recent_memories[i+1:], i+1):
                    try:
                        # Calculate semantic similarity
                        connection_strength, connection_type = await self._calculate_connection(
                            memory1, memory2
                        )
                        
                        # Only include connections above threshold
                        if connection_strength > self.connection_threshold:
                            connections.append({
                                "source": memory1.id,
                                "target": memory2.id,
                                "strength": connection_strength,
                                "type": connection_type
                            })
                            connection_strengths.append(connection_strength)
                            
                    except Exception as e:
                        logger.debug(f"Could not calculate connection between {memory1.id} and {memory2.id}: {e}")
                        continue
            
            # Calculate statistics
            total_nodes = len(nodes)
            total_connections = len(connections)
            avg_connections = (total_connections * 2) / max(total_nodes, 1)  # Each connection connects 2 nodes
            max_strength = max(connection_strengths) if connection_strengths else 0.0
            
            # Find most connected memory
            connection_counts = {}
            for conn in connections:
                connection_counts[conn["source"]] = connection_counts.get(conn["source"], 0) + 1
                connection_counts[conn["target"]] = connection_counts.get(conn["target"], 0) + 1
            
            most_connected = max(connection_counts.items(), key=lambda x: x[1])[0] if connection_counts else ""
            
            # Find the content of most connected memory
            most_connected_content = ""
            if most_connected:
                most_connected_memory = next((m for m in recent_memories if m.id == most_connected), None)
                if most_connected_memory:
                    most_connected_content = most_connected_memory.content[:50] + "..."
            
            stats = {
                "total_nodes": total_nodes,
                "total_connections": total_connections,
                "average_connections_per_node": avg_connections,
                "strongest_connection_strength": max_strength,
                "most_connected_memory": most_connected_content
            }
            
            logger.info(f"📊 Generated visualization data: {total_nodes} nodes, {total_connections} connections")
            
            return {
                "nodes": nodes,
                "connections": connections,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating visualization data: {e}")
            return {
                "nodes": [],
                "connections": [],
                "stats": {
                    "total_nodes": 0,
                    "total_connections": 0,
                    "average_connections_per_node": 0.0,
                    "strongest_connection_strength": 0.0,
                    "most_connected_memory": ""
                }
            } 