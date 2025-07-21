'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'

interface SemanticAnalysis {
  intent: string
  confidence: number
  reasoning: string
  emotional_tone: string
  memory_required: boolean
  processing_needs: Record<string, boolean>
}

interface ThinkingProcess {
  name: string
  status: 'idle' | 'active' | 'complete'
  description: string
  duration?: number
}

interface DashboardData {
  current_analysis?: SemanticAnalysis
  thinking_processes: ThinkingProcess[]
  memory_stats: {
    recent_recalls: number
    connections_explored: number
    semantic_matches: number
  }
  conversation_context: {
    message_count: number
    complexity_trend: 'increasing' | 'decreasing' | 'stable'
  }
}

export default function SemanticDashboard() {
  const [data, setData] = useState<DashboardData>({
    thinking_processes: [
      { name: 'Semantic Analysis', status: 'idle', description: 'Understanding message meaning and intent' },
      { name: 'Memory Recall', status: 'idle', description: 'Retrieving relevant contextual information' },
      { name: 'Planning Simulation', status: 'idle', description: 'Internal goal formulation and action simulation' },
      { name: 'Coherence Analysis', status: 'idle', description: 'Verifying response consistency' },
      { name: 'Background Processing', status: 'idle', description: 'Deep memory connections and insights' }
    ],
    memory_stats: {
      recent_recalls: 0,
      connections_explored: 0,
      semantic_matches: 0
    },
    conversation_context: {
      message_count: 0,
      complexity_trend: 'stable'
    }
  })

  const [isLive, setIsLive] = useState(false)

  useEffect(() => {
    // Simulate real-time updates (would connect to WebSocket in practice)
    const interval = setInterval(() => {
      if (isLive) {
        setData(prev => ({
          ...prev,
          memory_stats: {
            recent_recalls: prev.memory_stats.recent_recalls + Math.floor(Math.random() * 2),
            connections_explored: prev.memory_stats.connections_explored + Math.floor(Math.random() * 5),
            semantic_matches: prev.memory_stats.semantic_matches + Math.floor(Math.random() * 3)
          }
        }))
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [isLive])

  const getProcessStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success'
      case 'complete': return 'default'
      case 'idle': return 'secondary'
      default: return 'secondary'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success'
    if (confidence >= 0.6) return 'warning'
    return 'error'
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {/* Current Semantic Analysis */}
      <Card className="col-span-1 md:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            🧠 Current Semantic Analysis
            <Badge variant={isLive ? 'success' : 'secondary'}>
              {isLive ? 'Live' : 'Demo'}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {data.current_analysis ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className="font-medium">Intent:</span>
                <Badge variant="default">{data.current_analysis.intent}</Badge>
                <Badge variant={getConfidenceColor(data.current_analysis.confidence)}>
                  {(data.current_analysis.confidence * 100).toFixed(0)}% confidence
                </Badge>
              </div>
              
              <div>
                <span className="font-medium">Reasoning:</span>
                <p className="text-sm text-gray-600 mt-1">{data.current_analysis.reasoning}</p>
              </div>
              
              <div className="flex items-center gap-2">
                <span className="font-medium">Emotional Tone:</span>
                <Badge variant="secondary">{data.current_analysis.emotional_tone}</Badge>
                <span className="font-medium">Memory Required:</span>
                <Badge variant={data.current_analysis.memory_required ? 'success' : 'secondary'}>
                  {data.current_analysis.memory_required ? 'Yes' : 'No'}
                </Badge>
              </div>
            </div>
          ) : (
            <p className="text-gray-500">No active analysis - waiting for user input</p>
          )}
        </CardContent>
      </Card>

      {/* Thinking Processes */}
      <Card>
        <CardHeader>
          <CardTitle>🔄 Thinking Processes</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {data.thinking_processes.map((process, index) => (
              <div key={index} className="flex items-center justify-between p-2 border rounded">
                <div className="flex-1">
                  <div className="font-medium text-sm">{process.name}</div>
                  <div className="text-xs text-gray-500">{process.description}</div>
                </div>
                <Badge variant={getProcessStatusColor(process.status)}>
                  {process.status}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Memory Statistics */}
      <Card>
        <CardHeader>
          <CardTitle>🧠 Memory Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm">Recent Recalls:</span>
              <Badge variant="default">{data.memory_stats.recent_recalls}</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Connections Explored:</span>
              <Badge variant="success">{data.memory_stats.connections_explored}</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Semantic Matches:</span>
              <Badge variant="warning">{data.memory_stats.semantic_matches}</Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Conversation Context */}
      <Card>
        <CardHeader>
          <CardTitle>💬 Conversation Context</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm">Messages:</span>
              <Badge variant="default">{data.conversation_context.message_count}</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-sm">Complexity Trend:</span>
              <Badge variant={
                data.conversation_context.complexity_trend === 'increasing' ? 'warning' :
                data.conversation_context.complexity_trend === 'decreasing' ? 'success' : 'secondary'
              }>
                {data.conversation_context.complexity_trend}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Controls */}
      <Card className="col-span-1 md:col-span-2 lg:col-span-3">
        <CardHeader>
          <CardTitle>🎛️ Dashboard Controls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <button
              onClick={() => setIsLive(!isLive)}
              className={`px-4 py-2 rounded-lg font-medium ${
                isLive 
                  ? 'bg-red-500 text-white hover:bg-red-600' 
                  : 'bg-green-500 text-white hover:bg-green-600'
              }`}
            >
              {isLive ? '⏸️ Pause Live Updates' : '▶️ Start Live Updates'}
            </button>
            
            <button
              onClick={() => setData(prev => ({
                ...prev,
                current_analysis: {
                  intent: 'information_request',
                  confidence: 0.87,
                  reasoning: 'User is seeking specific factual information based on semantic analysis',
                  emotional_tone: 'curious',
                  memory_required: true,
                  processing_needs: { memory: true, planning: false, coherence: true }
                },
                thinking_processes: prev.thinking_processes.map(p => ({
                  ...p,
                  status: Math.random() > 0.5 ? 'active' : 'complete'
                }))
              }))}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 font-medium"
            >
              🧪 Simulate Analysis
            </button>
            
            <button
              onClick={() => setData(prev => ({
                thinking_processes: prev.thinking_processes.map(p => ({ ...p, status: 'idle' })),
                memory_stats: { recent_recalls: 0, connections_explored: 0, semantic_matches: 0 },
                conversation_context: { message_count: 0, complexity_trend: 'stable' },
                current_analysis: undefined
              }))}
              className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 font-medium"
            >
              🗑️ Clear Dashboard
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 