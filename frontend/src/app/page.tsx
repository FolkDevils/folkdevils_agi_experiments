'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, RefreshCw, Brain, Lightbulb } from 'lucide-react';
import { clsx } from 'clsx';
import SuggestionsPanel from '../components/SuggestionsPanel';
import MemoryViewer from '../components/MemoryViewer';
import MemoryGraphVisualization from '../components/MemoryGraphVisualization';
import RealtimeChat from '../components/RealtimeChat';
import SemanticDashboard from '../components/SemanticDashboard';
import { Button } from '../components/ui/button';

interface ChatMessage {
  id: string;
  message: string;
  isUser: boolean;
  timestamp: Date;
  responseTime?: number; // in seconds
}

interface ChatResponse {
  response: string;
  session_id: string;
  command_used?: string;
  reasoning?: string;
  memory_context?: string;
  voice_compliance_score?: number;
  timestamp: number;  // Unix timestamp for cache validation
  cache_version: string;  // Server version for cache invalidation
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [requestStartTime, setRequestStartTime] = useState<Date | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [showMemoryViewer, setShowMemoryViewer] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [showMemoryGraph, setShowMemoryGraph] = useState(false);
  const [showRealtimeChat, setShowRealtimeChat] = useState(false);
  const [showSemanticDashboard, setShowSemanticDashboard] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Cache validation function
  const validateCache = (messages: ChatMessage[]): boolean => {
    if (messages.length === 0) return true;
    
    try {
      // Check if we have cache metadata
      const cacheInfo = localStorage.getItem('son-of-andrew-cache-info');
      if (!cacheInfo) {
        console.log('No cache info found, clearing cache');
        return false;
      }
      
      const { lastTimestamp, cacheVersion } = JSON.parse(cacheInfo);
      const currentTime = Math.floor(Date.now() / 1000);
      const cacheAge = currentTime - lastTimestamp;
      
      // Invalidate cache if older than 10 minutes
      if (cacheAge > 10 * 60) {
        console.log('Cache expired (older than 10 minutes), clearing cache');
        return false;
      }
      
      // Invalidate cache if server version changed
      if (cacheVersion !== '2.0.0') {
        console.log('Server version changed, clearing cache');
        return false;
      }
      
      return true;
    } catch (error) {
      console.error('Cache validation error:', error);
      return false;
    }
  };

  // Clear cache helper
  const clearCache = () => {
    localStorage.removeItem('son-of-andrew-messages');
    localStorage.removeItem('son-of-andrew-cache-info');
    localStorage.removeItem('son-of-andrew-session-id');
  };

  // Load session from localStorage on mount
  useEffect(() => {
    const savedSessionId = localStorage.getItem('son-of-andrew-session-id');
    const savedMessages = localStorage.getItem('son-of-andrew-messages');
    
    if (savedMessages) {
      try {
        const parsed = JSON.parse(savedMessages) as Array<{
          id: string;
          message: string;
          isUser: boolean;
          timestamp: string;
          responseTime?: number;
        }>;
        const parsedMessages = parsed.map((msg) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        
        // Validate cache before loading
        if (validateCache(parsedMessages)) {
          console.log('Cache validated successfully, loading messages');
          setMessages(parsedMessages);
          if (savedSessionId) {
            setSessionId(savedSessionId);
          }
        } else {
          console.log('Cache validation failed, starting fresh');
          clearCache();
        }
      } catch (error) {
        console.error('Failed to parse saved messages:', error);
        clearCache();
      }
    } else if (savedSessionId) {
      setSessionId(savedSessionId);
    }
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('son-of-andrew-messages', JSON.stringify(messages));
      // Update cache metadata
      const cacheInfo = {
        lastTimestamp: Math.floor(Date.now() / 1000),
        cacheVersion: '2.0.0'
      };
      localStorage.setItem('son-of-andrew-cache-info', JSON.stringify(cacheInfo));
    }
  }, [messages]);

  // Save session ID to localStorage whenever it changes
  useEffect(() => {
    if (sessionId) {
      localStorage.setItem('son-of-andrew-session-id', sessionId);
    }
  }, [sessionId]);

  // Auto-resize textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [inputMessage]);

  // Timer effect for tracking response time
  useEffect(() => {
    if (isLoading && requestStartTime) {
      // Start immediately to show 0s, then update every second
      const updateTimer = () => {
        const now = new Date();
        const elapsed = Math.floor((now.getTime() - requestStartTime.getTime()) / 1000);
        setElapsedTime(elapsed);
      };
      
      updateTimer(); // Initial call
      timerRef.current = setInterval(updateTimer, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isLoading, requestStartTime]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      message: inputMessage.trim(),
      isUser: true,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    
    const startTime = new Date();
    setRequestStartTime(startTime);
    setElapsedTime(0);
    setIsLoading(true);

    try {
      // Build conversation context from recent messages
      const conversationContext = messages.slice(-10).map(msg => ({
        role: msg.isUser ? 'user' : 'assistant',
        content: msg.message,
        timestamp: msg.timestamp.toISOString()
      }));

      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.message,
          speaker: 'andrew',
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Calculate response time
      const endTime = new Date();
      const responseTimeSeconds = requestStartTime ? 
        Math.floor((endTime.getTime() - requestStartTime.getTime()) / 1000) : 0;

      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: data.response,
        isUser: false,
        timestamp: endTime,
        responseTime: responseTimeSeconds,
      };

      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      console.error('Failed to send message:', error);
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: 'Sorry, I encountered an error. Please try again.',
        isUser: false,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setRequestStartTime(null);
      setElapsedTime(0);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(null);
    clearCache();
  };

  const handleSuggestionSelect = (suggestion: any) => {
    // When a suggestion is accepted, add it as a message to continue the conversation
    setInputMessage(suggestion.content);
    setShowSuggestions(false);
    
    // Optionally auto-send the suggestion
    // Or just populate the input field for the user to review and send
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 font-roboto">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center">
              <span className="text-white font-medium text-lg">SA</span>
            </div>
            <div>
              <h1 className="text-xl font-medium text-gray-900">Son of Andrew</h1>
              {sessionId && (
                <p className="text-sm text-gray-500">Session: {sessionId.slice(0, 8)}...</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowMemoryViewer(!showMemoryViewer)}
              className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <Brain className="w-4 h-4" />
              {showMemoryViewer ? 'Hide' : 'Show'} Memory Viewer
            </button>
            <button
              onClick={() => setShowMemoryGraph(!showMemoryGraph)}
              className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <Brain className="w-4 h-4" />
              {showMemoryGraph ? 'Hide' : 'Show'} Memory Graph
            </button>
            <button
              onClick={() => setShowSuggestions(true)}
              className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <Lightbulb className="w-4 h-4" />
              Suggestions
            </button>
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                New Chat
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Chat Messages */}
        <div className={clsx("flex-1 overflow-y-auto px-4 py-6", showMemoryViewer && "hidden md:block")}>
          <div className="max-w-4xl mx-auto space-y-4">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full min-h-[400px]">
                <div className="text-center">
                  <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="text-3xl">🤖</span>
                  </div>
                  <h2 className="text-2xl font-medium text-gray-800 mb-2">Welcome to Son of Andrew</h2>
                  <p className="text-gray-600">Send a message to start the conversation</p>
                </div>
              </div>
            ) : (
              <>
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={clsx(
                      'flex',
                      message.isUser ? 'justify-end' : 'justify-start'
                    )}
                  >
                    <div
                      className={clsx(
                        'max-w-xs md:max-w-md lg:max-w-lg px-4 py-3 rounded-2xl shadow-sm',
                        message.isUser
                          ? 'bg-blue-500 text-white rounded-br-md'
                          : 'bg-white text-gray-800 rounded-bl-md border border-gray-200'
                      )}
                    >
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.message}</p>
                      <p
                        className={clsx(
                          'text-xs mt-2 opacity-70',
                          message.isUser ? 'text-blue-100' : 'text-gray-500'
                        )}
                      >
                        {message.timestamp.toLocaleTimeString([], { 
                          hour: '2-digit', 
                          minute: '2-digit' 
                        })}
                        {!message.isUser && message.responseTime && (
                          <span className="ml-2">• {message.responseTime}s response</span>
                        )}
                      </p>
                    </div>
                  </div>
                ))}
                
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-white text-gray-800 max-w-xs md:max-w-md lg:max-w-lg px-4 py-3 rounded-2xl rounded-bl-md shadow-sm border border-gray-200">
                      <div className="flex items-center space-x-2">
                        <div className="flex items-center justify-center w-8 h-8 bg-blue-100 rounded-full">
                          <span className="text-sm font-mono text-blue-600">{elapsedTime}s</span>
                        </div>
                        <span className="text-xs text-gray-500">Son of Andrew is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </>
            )}
          </div>
        </div>
        
        {/* Memory Viewer Placeholder */}
        {showMemoryViewer && (
          <div className="w-full md:w-1/2 lg:w-2/3 border-l border-gray-200 overflow-y-auto">
            <MemoryViewer />
          </div>
        )}
        
        {/* Memory Graph Visualization */}
        {showMemoryGraph && (
          <div className="w-full md:w-1/2 lg:w-2/3 border-l border-gray-200 overflow-y-auto">
            <div className="p-4">
              <MemoryGraphVisualization />
            </div>
          </div>
        )}

        {/* Dashboard Toggle */}
        <div className="flex gap-4 mb-6">
          <Button
            onClick={() => setShowSemanticDashboard(!showSemanticDashboard)}
            variant="outline"
            className="flex items-center gap-2"
          >
            📊 {showSemanticDashboard ? 'Hide' : 'Show'} Semantic Dashboard
          </Button>
        </div>

        {/* Semantic Dashboard */}
        {showSemanticDashboard && (
          <div className="mb-8">
            <SemanticDashboard />
          </div>
        )}

        {/* Real-time Chat Toggle */}
        <div className="flex gap-4 mb-6">
          <Button
            onClick={() => setShowRealtimeChat(!showRealtimeChat)}
            variant="outline"
            className="flex items-center gap-2"
          >
            🚀 {showRealtimeChat ? 'Hide' : 'Show'} Real-time Chat
          </Button>
        </div>

        {/* Real-time Chat Component */}
        {showRealtimeChat && (
          <div className="mb-8">
            <RealtimeChat />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-200 px-4 py-4 pb-4">
        <div className="mx-auto" style={{ width: 'max(33.33vw, 800px)' }}>
          <div className="flex items-end gap-3 bg-gray-100 rounded-2xl px-4 py-2">
            <textarea
              ref={textareaRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type a message..."
              className="flex-1 bg-transparent text-gray-800 placeholder-gray-500 min-h-[3rem] max-h-32 focus:outline-none text-sm resize-none overflow-y-auto"
              disabled={isLoading}
              rows={1}
            />
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className={clsx(
                'w-8 h-8 rounded-full flex items-center justify-center transition-colors flex-shrink-0 mb-1',
                !inputMessage.trim() || isLoading
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              )}
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2 text-center">
            Press Enter to send • Shift+Enter for new line
          </p>
        </div>
      </div>

      {/* Suggestions Panel */}
      <SuggestionsPanel
        isVisible={showSuggestions}
        onClose={() => setShowSuggestions(false)}
        onSuggestionSelect={handleSuggestionSelect}
      />
    </div>
  );
} 