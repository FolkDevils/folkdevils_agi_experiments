'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'

interface Message {
  id: string
  text: string
  sender: 'user' | 'ai'
  timestamp: Date
  semanticAnalysis?: {
    intent: string
    confidence: number
    reasoning: string
    emotional_tone: string
    memory_required: boolean
  }
}

interface ThinkingState {
  isThinking: boolean
  status: string
  semanticAnalysis?: any
}

export default function RealtimeChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isConnected, setIsConnected] = useState(false)
  const [thinking, setThinking] = useState<ThinkingState>({ isThinking: false, status: '' })
  const [conversationId] = useState(() => `conv_${Date.now()}`)
  
  const wsRef = useRef<WebSocket | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(scrollToBottom, [messages])

  const connectWebSocket = useCallback(() => {
    try {
      const ws = new WebSocket(`ws://localhost:8000/ws/${conversationId}`)
      
      ws.onopen = () => {
        console.log('🔗 WebSocket connected')
        setIsConnected(true)
        setThinking({ isThinking: false, status: 'Connected to consciousness' })
      }
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        
        switch (data.type) {
          case 'connection_established':
            console.log('✅ Consciousness connection established')
            break
            
          case 'thinking_started':
            setThinking({ 
              isThinking: true, 
              status: 'AI is analyzing your message...' 
            })
            break
            
          case 'semantic_analysis':
            setThinking({ 
              isThinking: true, 
              status: `Understanding: ${data.data.intent} (${(data.data.confidence * 100).toFixed(0)}% confidence)`,
              semanticAnalysis: data.data
            })
            break
            
          case 'thinking_completed':
            setThinking({ isThinking: false, status: 'Complete' })
            
            // Add AI response to messages
            const aiMessage: Message = {
              id: `ai_${Date.now()}`,
              text: data.response,
              sender: 'ai',
              timestamp: new Date(),
              semanticAnalysis: thinking.semanticAnalysis
            }
            setMessages(prev => [...prev, aiMessage])
            break
            
          case 'thinking_error':
            setThinking({ 
              isThinking: false, 
              status: `Error: ${data.error}` 
            })
            break
            
          default:
            console.log('Received WebSocket message:', data)
        }
      }
      
      ws.onclose = () => {
        console.log('🔌 WebSocket disconnected')
        setIsConnected(false)
        setThinking({ isThinking: false, status: 'Disconnected' })
        
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000)
      }
      
      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error)
        setIsConnected(false)
      }
      
      wsRef.current = ws
      
    } catch (error) {
      console.error('❌ Failed to connect WebSocket:', error)
      setIsConnected(false)
    }
  }, [conversationId, thinking.semanticAnalysis])

  useEffect(() => {
    connectWebSocket()
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connectWebSocket])

  const sendMessage = async () => {
    if (!inputMessage.trim()) return

    // Add user message to chat
    const userMessage: Message = {
      id: `user_${Date.now()}`,
      text: inputMessage,
      sender: 'user',
      timestamp: new Date()
    }
    setMessages(prev => [...prev, userMessage])

    // Send to backend via HTTP (which will trigger WebSocket notifications)
    try {
      const response = await fetch('http://localhost:8000/api/chat/realtime', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: inputMessage,
          speaker: 'andrew',
          conversation_id: conversationId
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      // Response will come via WebSocket
      setInputMessage('')
      
    } catch (error: any) {
      console.error('❌ Error sending message:', error)
      setThinking({ 
        isThinking: false, 
        status: `Error: ${error?.message || 'Unknown error'}` 
      })
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <Card className="w-full max-w-4xl mx-auto h-[600px] flex flex-col">
      <CardHeader className="flex-shrink-0">
        <CardTitle className="flex items-center gap-2">
          🧠 Real-time Conversational Consciousness
          <Badge variant={isConnected ? "default" : "secondary"}>
            {isConnected ? '🔗 Connected' : '🔌 Disconnected'}
          </Badge>
        </CardTitle>
        
        {/* Thinking Indicator */}
        {thinking.isThinking && (
          <div className="flex items-center gap-2 p-2 bg-blue-50 rounded-lg border border-blue-200">
            <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
            <span className="text-sm text-blue-700">{thinking.status}</span>
            {thinking.semanticAnalysis && (
                                   <Badge variant="secondary" className="text-xs">
                {thinking.semanticAnalysis.emotional_tone}
              </Badge>
            )}
          </div>
        )}
      </CardHeader>
      
      <CardContent className="flex-1 flex flex-col gap-4 overflow-hidden">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto space-y-3 p-2">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] p-3 rounded-lg ${
                  message.sender === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                <p className="text-sm">{message.text}</p>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs opacity-70">
                    {message.timestamp.toLocaleTimeString()}
                  </span>
                  {message.semanticAnalysis && (
                    <Badge variant="secondary" className="text-xs">
                      {message.semanticAnalysis.intent}
                    </Badge>
                  )}
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input Area */}
        <div className="flex-shrink-0 flex gap-2">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything..."
            className="flex-1 p-3 border border-gray-300 rounded-lg resize-none"
            rows={2}
            disabled={!isConnected || thinking.isThinking}
          />
          <Button 
            onClick={sendMessage}
            disabled={!isConnected || thinking.isThinking || !inputMessage.trim()}
            className="self-end"
          >
            Send
          </Button>
        </div>
      </CardContent>
    </Card>
  )
} 