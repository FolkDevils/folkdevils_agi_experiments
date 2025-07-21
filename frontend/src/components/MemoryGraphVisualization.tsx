"use client";

import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

interface MemoryNode {
  id: string;
  content: string;
  type: 'episodic' | 'semantic' | 'identity' | 'relationship';
  importance: number;
  emotional_weight: number;
  timestamp: string;
  x?: number;
  y?: number;
}

interface MemoryConnection {
  source: string;
  target: string;
  strength: number;
  type: 'semantic' | 'temporal' | 'causal' | 'associative' | 'emotional';
}

interface MemoryGraphData {
  nodes: MemoryNode[];
  connections: MemoryConnection[];
  graph_stats: {
    total_nodes: number;
    total_connections: number;
    average_connections_per_node: number;
    strongest_connection_strength: number;
    most_connected_memory: string;
  };
}

export default function MemoryGraphVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [graphData, setGraphData] = useState<MemoryGraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<MemoryNode | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showConnectionTypes, setShowConnectionTypes] = useState(true);

  // Canvas dimensions
  const CANVAS_WIDTH = 800;
  const CANVAS_HEIGHT = 600;
  const NODE_RADIUS = 8;
  const MAX_NODE_RADIUS = 20;

  useEffect(() => {
    loadMemoryGraph();
  }, []);

  useEffect(() => {
    if (graphData && canvasRef.current) {
      drawGraph();
    }
  }, [graphData, showConnectionTypes]);

  const loadMemoryGraph = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:8000/api/memory/graph');
      
      if (!response.ok) {
        throw new Error(`Failed to load memory graph: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Position nodes using a simple force-directed layout simulation
      const positionedData = positionNodes(data);
      setGraphData(positionedData);
      setError(null);
    } catch (err) {
      console.error('Error loading memory graph:', err);
      setError(err instanceof Error ? err.message : 'Failed to load memory graph');
    } finally {
      setIsLoading(false);
    }
  };

  const positionNodes = (data: MemoryGraphData): MemoryGraphData => {
    const nodes = [...data.nodes];
    const connections = data.connections;
    
    // Simple circular layout with some clustering
    const centerX = CANVAS_WIDTH / 2;
    const centerY = CANVAS_HEIGHT / 2;
    const radius = Math.min(CANVAS_WIDTH, CANVAS_HEIGHT) * 0.35;
    
    // Group nodes by type
    const nodesByType: { [key: string]: MemoryNode[] } = {
      episodic: [],
      semantic: [],
      identity: [],
      relationship: []
    };
    
    nodes.forEach(node => {
      nodesByType[node.type].push(node);
    });
    
    // Position nodes in type-based clusters
    const typeAngles = {
      episodic: 0,
      semantic: Math.PI / 2,
      identity: Math.PI,
      relationship: (3 * Math.PI) / 2
    };
    
    Object.entries(nodesByType).forEach(([type, typeNodes]) => {
      const baseAngle = typeAngles[type as keyof typeof typeAngles];
      const angleSpread = Math.PI / 3; // 60 degrees spread per type
      
      typeNodes.forEach((node, index) => {
        const angle = baseAngle + (angleSpread * (index / Math.max(typeNodes.length - 1, 1)) - angleSpread / 2);
        const nodeRadius = radius * (0.7 + 0.3 * node.importance); // Vary distance by importance
        
        node.x = centerX + Math.cos(angle) * nodeRadius;
        node.y = centerY + Math.sin(angle) * nodeRadius;
      });
    });
    
    return { ...data, nodes };
  };

  const drawGraph = () => {
    const canvas = canvasRef.current;
    if (!canvas || !graphData) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // Draw connections first (so they appear behind nodes)
    if (showConnectionTypes) {
      drawConnections(ctx);
    }
    
    // Draw nodes
    drawNodes(ctx);
    
    // Draw legend
    drawLegend(ctx);
  };

  const drawConnections = (ctx: CanvasRenderingContext2D) => {
    if (!graphData) return;
    
    const connectionColors = {
      semantic: '#3B82F6',     // Blue
      temporal: '#10B981',     // Green  
      causal: '#F59E0B',       // Orange
      associative: '#8B5CF6',  // Purple
      emotional: '#EF4444'     // Red
    };
    
    graphData.connections.forEach(connection => {
      const sourceNode = graphData.nodes.find(n => n.id === connection.source);
      const targetNode = graphData.nodes.find(n => n.id === connection.target);
      
      if (sourceNode && targetNode && sourceNode.x !== undefined && sourceNode.y !== undefined && 
          targetNode.x !== undefined && targetNode.y !== undefined) {
        
        ctx.strokeStyle = connectionColors[connection.type] || '#6B7280';
        ctx.lineWidth = Math.max(1, connection.strength * 3);
        ctx.globalAlpha = 0.3 + (connection.strength * 0.4);
        
        ctx.beginPath();
        ctx.moveTo(sourceNode.x, sourceNode.y);
        ctx.lineTo(targetNode.x, targetNode.y);
        ctx.stroke();
        
        ctx.globalAlpha = 1;
      }
    });
  };

  const drawNodes = (ctx: CanvasRenderingContext2D) => {
    if (!graphData) return;
    
    const nodeColors = {
      episodic: '#3B82F6',     // Blue
      semantic: '#10B981',     // Green
      identity: '#F59E0B',     // Orange
      relationship: '#EF4444'  // Red
    };
    
    graphData.nodes.forEach(node => {
      if (node.x === undefined || node.y === undefined) return;
      
      // Calculate node size based on importance and emotional weight
      const size = NODE_RADIUS + (node.importance * node.emotional_weight * (MAX_NODE_RADIUS - NODE_RADIUS));
      
      // Draw node
      ctx.fillStyle = nodeColors[node.type];
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw border if selected
      if (selectedNode && selectedNode.id === node.id) {
        ctx.strokeStyle = '#1F2937';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
      
      ctx.globalAlpha = 1;
    });
  };

  const drawLegend = (ctx: CanvasRenderingContext2D) => {
    const legendX = 10;
    const legendY = 10;
    
    // Memory types legend
    ctx.fillStyle = '#1F2937';
    ctx.font = '12px sans-serif';
    ctx.fillText('Memory Types:', legendX, legendY + 15);
    
    const nodeColors = {
      episodic: '#3B82F6',
      semantic: '#10B981', 
      identity: '#F59E0B',
      relationship: '#EF4444'
    };
    
    Object.entries(nodeColors).forEach(([type, color], index) => {
      const y = legendY + 35 + (index * 20);
      
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(legendX + 8, y, 6, 0, 2 * Math.PI);
      ctx.fill();
      
      ctx.fillStyle = '#1F2937';
      ctx.fillText(type, legendX + 20, y + 4);
    });
    
    // Connection types legend (if enabled)
    if (showConnectionTypes) {
      ctx.fillText('Connection Types:', legendX, legendY + 140);
      
      const connectionColors = {
        semantic: '#3B82F6',
        temporal: '#10B981',
        causal: '#F59E0B', 
        associative: '#8B5CF6',
        emotional: '#EF4444'
      };
      
      Object.entries(connectionColors).forEach(([type, color], index) => {
        const y = legendY + 160 + (index * 15);
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(legendX, y);
        ctx.lineTo(legendX + 15, y);
        ctx.stroke();
        
        ctx.fillStyle = '#1F2937';
        ctx.fillText(type, legendX + 20, y + 4);
      });
    }
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!graphData) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Find clicked node
    const clickedNode = graphData.nodes.find(node => {
      if (node.x === undefined || node.y === undefined) return false;
      
      const distance = Math.sqrt(Math.pow(x - node.x, 2) + Math.pow(y - node.y, 2));
      const nodeSize = NODE_RADIUS + (node.importance * node.emotional_weight * (MAX_NODE_RADIUS - NODE_RADIUS));
      
      return distance <= nodeSize;
    });
    
    setSelectedNode(clickedNode || null);
  };

  const getNodeConnections = (nodeId: string) => {
    if (!graphData) return [];
    
    return graphData.connections.filter(conn => 
      conn.source === nodeId || conn.target === nodeId
    );
  };

  if (isLoading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>🧠 Memory Graph</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-500">Loading memory graph...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>🧠 Memory Graph</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p className="text-red-600 mb-4">Error: {error}</p>
            <Button onClick={loadMemoryGraph} variant="outline">
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!graphData) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>🧠 Memory Graph</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-500 text-center py-8">No memory graph data available</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="w-full space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            🧠 Memory Graph
            <div className="flex gap-2">
              <Button
                variant={showConnectionTypes ? "default" : "outline"}
                size="sm"
                onClick={() => setShowConnectionTypes(!showConnectionTypes)}
              >
                Connections
              </Button>
              <Button onClick={loadMemoryGraph} variant="outline" size="sm">
                Refresh
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Graph Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{graphData.graph_stats.total_nodes}</div>
              <div className="text-sm text-gray-500">Memories</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{graphData.graph_stats.total_connections}</div>
              <div className="text-sm text-gray-500">Connections</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {graphData.graph_stats.average_connections_per_node.toFixed(1)}
              </div>
              <div className="text-sm text-gray-500">Avg Connections</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {(graphData.graph_stats.strongest_connection_strength * 100).toFixed(0)}%
              </div>
              <div className="text-sm text-gray-500">Max Strength</div>
            </div>
          </div>
          
          {/* Graph Canvas */}
          <div className="border rounded-lg overflow-hidden">
            <canvas
              ref={canvasRef}
              width={CANVAS_WIDTH}
              height={CANVAS_HEIGHT}
              onClick={handleCanvasClick}
              className="cursor-pointer"
              style={{ width: '100%', height: 'auto' }}
            />
          </div>
          
          <p className="text-sm text-gray-500 mt-2">
            Click on nodes to view details. Different colors represent different memory types.
          </p>
        </CardContent>
      </Card>
      
      {/* Selected Node Details */}
      {selectedNode && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Memory Details
              <Badge variant="secondary">{selectedNode.type}</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div>
                <strong>Content:</strong>
                <p className="text-sm text-gray-700 mt-1">{selectedNode.content}</p>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <strong>Importance:</strong>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${selectedNode.importance * 100}%` }}
                      />
                    </div>
                    <span className="text-sm">{(selectedNode.importance * 100).toFixed(0)}%</span>
                  </div>
                </div>
                
                <div>
                  <strong>Emotional Weight:</strong>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-red-600 h-2 rounded-full" 
                        style={{ width: `${selectedNode.emotional_weight * 100}%` }}
                      />
                    </div>
                    <span className="text-sm">{(selectedNode.emotional_weight * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
              
              <div>
                <strong>Timestamp:</strong>
                <p className="text-sm text-gray-600">{new Date(selectedNode.timestamp).toLocaleString()}</p>
              </div>
              
              <div>
                <strong>Connections ({getNodeConnections(selectedNode.id).length}):</strong>
                <div className="flex flex-wrap gap-1 mt-1">
                  {getNodeConnections(selectedNode.id).map((conn, index) => (
                    <Badge key={index} variant="secondary" className="text-xs">
                      {conn.type} ({(conn.strength * 100).toFixed(0)}%)
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
} 