'use client';

import { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Send, RefreshCw, Brain, Lightbulb, Volume2, Settings, Activity } from 'lucide-react';
import { clsx } from 'clsx';
import SuggestionsPanel from '../components/SuggestionsPanel';
import MemoryViewer from '../components/MemoryViewer';
import MemoryGraphVisualization from '../components/MemoryGraphVisualization';
import RealtimeChat from '../components/RealtimeChat';
import SemanticDashboard from '../components/SemanticDashboard';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';

interface ChatMessage {
  id: string;
  message: string;
  isUser: boolean;
  timestamp: Date;
  responseTime?: number; // in seconds
  isVoice?: boolean; // Whether message was voice input/output
  traceId?: string; // For performance tracking
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
  trace_id?: string;
  performance?: {
    total_time: number;
    semantic_time: number;
    parallel_time: number;
    response_time: number;
    coherence_time: number;
    coherence_skipped: boolean;
    intent_confidence: number;
    memory_lookup_required: boolean;
    memories_recalled: number;
  };
  semantic_analysis?: {
    intent: string;
    confidence: number;
    reasoning: string;
    emotional_tone: string;
    response_type: string;
  };
}

interface VoiceState {
  isListening: boolean;
  isProcessing: boolean;
  isSpeaking: boolean;
  audioLevel: number;
  error?: string;
}

export default function ChatPage() {
  // Generate unique IDs for messages
  const idCounterRef = useRef<number>(0);
  const generateUniqueId = () => {
    idCounterRef.current += 1;
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}-${performance.now()}-${idCounterRef.current}`;
  };

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [requestStartTime, setRequestStartTime] = useState<Date | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  
  // Voice-specific state
  const [voiceState, setVoiceState] = useState<VoiceState>({
    isListening: false,
    isProcessing: false,
    isSpeaking: false,
    audioLevel: 0
  });
  const [isVoiceMode, setIsVoiceMode] = useState(true); // Default to voice-first
  const [currentPerformance, setCurrentPerformance] = useState<ChatResponse['performance'] | null>(null);
  const [currentSemanticAnalysis, setCurrentSemanticAnalysis] = useState<ChatResponse['semantic_analysis'] | null>(null);
  const [lastProcessedTranscript, setLastProcessedTranscript] = useState<string>('');
  
  // UI state
  const [showMemoryViewer, setShowMemoryViewer] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [showMemoryGraph, setShowMemoryGraph] = useState(false);
  const [showRealtimeChat, setShowRealtimeChat] = useState(false);
  const [showSemanticDashboard, setShowSemanticDashboard] = useState(true); // Show by default for consciousness demo
  const [showPerformanceMetrics, setShowPerformanceMetrics] = useState(true); // Show performance by default
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioWebSocketRef = useRef<WebSocket | null>(null);
  const recognitionRef = useRef<any>(null);

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
      if (cacheVersion !== '2.1.0') {
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
          id: msg.id && msg.id.includes('-') ? msg.id : generateUniqueId(), // Regenerate ID if it looks old/problematic
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
        cacheVersion: '2.1.0'
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

  // WebSocket connection for audio streaming
  useEffect(() => {
    if (isVoiceMode) {
      const conversationId = sessionId || `conv_${Date.now()}`;
      const wsUrl = `ws://localhost:8000/ws/audio/${conversationId}`;
      
      try {
        const audioWs = new WebSocket(wsUrl);
        audioWebSocketRef.current = audioWs;
        
        audioWs.onopen = () => {
          console.log('🎤 Audio WebSocket connected');
          audioWs.send(JSON.stringify({
            type: "start_voice_session",
            timestamp: new Date().toISOString()
          }));
        };
        
        audioWs.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('🎤 Audio WebSocket message:', data.type);
            
            if (data.type === "consciousness_response") {
              // Handle consciousness response from voice input
              const responseData = data.data;
              
              const aiMessage: ChatMessage = {
                id: generateUniqueId(),
                message: responseData.response,
                isUser: false,
                timestamp: new Date(),
                responseTime: responseData.processing_time,
                traceId: responseData.trace_id,
                isVoice: true,
              };
              
              // Add AI response to messages
              setMessages(prev => [...prev, aiMessage]);
              
              // Update performance and semantic data
              if (responseData.semantic_analysis) {
                setCurrentSemanticAnalysis(responseData.semantic_analysis);
              }
              
              // Set performance data if available
              const performanceData = {
                total_time: responseData.processing_time,
                semantic_time: responseData.processing_time * 0.5, // Estimate
                parallel_time: 0,
                response_time: responseData.processing_time * 0.4,
                coherence_time: 0,
                coherence_skipped: true,
                intent_confidence: responseData.semantic_analysis?.confidence || 0.8,
                memory_lookup_required: responseData.memories_recalled > 0,
                memories_recalled: responseData.memories_recalled || 0
              };
              setCurrentPerformance(performanceData);
              
              // Speak the response
              if (responseData.response) {
                speakText(responseData.response);
              }
              
              // Mark voice processing as complete
              setVoiceState(prev => ({ ...prev, isProcessing: false }));
            }
            
          } catch (error) {
            console.error('🎤 Error parsing audio WebSocket message:', error);
          }
        };
        
        audioWs.onerror = (error) => {
          console.error('🎤 Audio WebSocket error:', error);
          setVoiceState(prev => ({ 
            ...prev, 
            error: 'Voice connection failed' 
          }));
        };
        
        audioWs.onclose = () => {
          console.log('🎤 Audio WebSocket disconnected');
        };
        
      } catch (error) {
        console.error('🎤 Failed to connect audio WebSocket:', error);
        setVoiceState(prev => ({ 
          ...prev, 
          error: 'Failed to establish voice connection' 
        }));
      }
    }
    
    // Cleanup WebSocket when voice mode is disabled or component unmounts
    return () => {
      if (audioWebSocketRef.current) {
        audioWebSocketRef.current.close();
        audioWebSocketRef.current = null;
      }
    };
  }, [isVoiceMode, sessionId]);

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

  const sendMessage = async (messageText?: string, isVoiceInput: boolean = false) => {
    const messageToSend = messageText || inputMessage.trim();
    if (!messageToSend || isLoading) return;

    const userMessage: ChatMessage = {
      id: generateUniqueId(),
      message: messageToSend,
      isUser: true,
      timestamp: new Date(),
      isVoice: isVoiceInput,
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    
    const startTime = new Date();
    setRequestStartTime(startTime);
    setElapsedTime(0);
    setIsLoading(true);

    try {
      // Use the new streaming endpoint for better performance
      const response = await fetch('http://localhost:8000/api/chat/stream', {
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
        id: generateUniqueId(),
        message: data.response,
        isUser: false,
        timestamp: endTime,
        responseTime: responseTimeSeconds,
        traceId: data.trace_id,
        isVoice: isVoiceMode,
      };

      setMessages(prev => [...prev, aiMessage]);

      // Update performance and semantic analysis data for display
      setCurrentPerformance(data.performance);
      setCurrentSemanticAnalysis(data.semantic_analysis);

      // If voice mode and response received, speak it
      if (isVoiceMode && data.response) {
        await speakText(data.response);
      }

    } catch (error) {
      console.error('Failed to send message:', error);
      
      const errorMessage: ChatMessage = {
        id: generateUniqueId(),
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

  // Enhanced voice-to-text functionality using Web Speech API + WebSocket
  const startListening = async () => {
    try {
      setVoiceState(prev => ({ ...prev, isListening: true, error: undefined }));
      setLastProcessedTranscript(''); // Clear previous transcript to allow new ones

      // Check if Web Speech API is available
      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        // Use Web Speech API for better real-time transcription
        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';
        
        recognitionRef.current = recognition;
        
        recognition.onstart = () => {
          console.log('🎤 Speech recognition started');
          setVoiceState(prev => ({ ...prev, isListening: true }));
        };
        
        recognition.onresult = (event: any) => {
          let finalTranscript = '';
          let interimTranscript = '';
          
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += transcript;
            } else {
              interimTranscript += transcript;
            }
          }
          
          // Show interim results in input
          if (interimTranscript) {
            setInputMessage(interimTranscript);
          }
          
          // Process final transcript
          if (finalTranscript.trim()) {
            console.log('🎤 Final transcript:', finalTranscript);
            processVoiceTranscript(finalTranscript.trim(), event.results[event.results.length - 1][0].confidence);
          }
        };
        
        recognition.onerror = (event: any) => {
          console.error('🎤 Speech recognition error:', event.error);
          setVoiceState(prev => ({ 
            ...prev, 
            isListening: false, 
            error: `Speech recognition error: ${event.error}` 
          }));
        };
        
        recognition.onend = () => {
          console.log('🎤 Speech recognition ended');
          setVoiceState(prev => ({ ...prev, isListening: false }));
        };
        
        recognition.start();
        
      } else {
        // Fallback to MediaRecorder approach
        await startMediaRecorderListening();
      }

      // Always set up audio visualization
      await setupAudioVisualization();

    } catch (error) {
      console.error('Error starting voice recording:', error);
      setVoiceState(prev => ({ 
        ...prev, 
        isListening: false, 
        error: 'Microphone access denied or not available' 
      }));
    }
  };

  const setupAudioVisualization = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Set up audio analysis for visualization
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      const microphone = audioContext.createMediaStreamSource(stream);
      microphone.connect(analyser);
      analyser.fftSize = 256;

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      // Start audio level monitoring
      const updateAudioLevel = () => {
        if (analyserRef.current && voiceState.isListening) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
          setVoiceState(prev => ({ ...prev, audioLevel: average / 255 }));
          animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
        }
      };
      updateAudioLevel();
      
    } catch (error) {
      console.warn('Audio visualization setup failed:', error);
    }
  };

  const startMediaRecorderListening = async () => {
    // Fallback MediaRecorder implementation
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const audioChunks: Blob[] = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      
      // For demo - simulate transcription
      const demoText = "Voice input received (MediaRecorder fallback). In production, this would be processed by OpenAI Whisper.";
      await processVoiceTranscript(demoText, 0.8);
      
      // Clean up
      stream.getTracks().forEach(track => track.stop());
    };

    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    
    if (mediaRecorderRef.current && voiceState.isListening) {
      mediaRecorderRef.current.stop();
    }
    
    // Clean up audio visualization
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    setVoiceState(prev => ({ ...prev, isListening: false, isProcessing: true }));
  };

  const processVoiceTranscript = async (transcript: string, confidence: number) => {
    try {
      console.log('🎤 Processing voice transcript:', transcript);
      
      // Prevent duplicate processing of the same transcript
      if (transcript === lastProcessedTranscript) {
        console.log('🎤 Duplicate transcript ignored:', transcript);
        return;
      }
      
      setLastProcessedTranscript(transcript);
      setVoiceState(prev => ({ ...prev, isProcessing: true }));
      
      // Add user message to chat with unique ID
      const userMessage: ChatMessage = {
        id: generateUniqueId(),
        message: transcript,
        isUser: true,
        timestamp: new Date(),
        isVoice: true,
      };
      
      setMessages(prev => [...prev, userMessage]);
      setInputMessage('');
      
      // Send to consciousness via WebSocket (primary path for voice)
      if (audioWebSocketRef.current && audioWebSocketRef.current.readyState === WebSocket.OPEN) {
        console.log('🎤 Sending via WebSocket');
        audioWebSocketRef.current.send(JSON.stringify({
          type: "transcribed_text",
          text: transcript,
          confidence: confidence,
          timestamp: new Date().toISOString()
        }));
        // Don't call sendMessage - WebSocket will handle the response
      } else {
        // Fallback to HTTP endpoint only if WebSocket unavailable
        console.warn('🎤 WebSocket unavailable, falling back to HTTP');
        // Remove the user message we just added since sendMessage will add it again
        setMessages(prev => prev.slice(0, -1));
        await sendMessage(transcript, true);
      }
      
      setVoiceState(prev => ({ ...prev, isProcessing: false }));
      
    } catch (error) {
      console.error('Error processing voice transcript:', error);
      setVoiceState(prev => ({ 
        ...prev, 
        isProcessing: false, 
        error: 'Failed to process voice input' 
      }));
    }
  };

  const speakText = async (text: string) => {
    try {
      setVoiceState(prev => ({ ...prev, isSpeaking: true }));
      
      // Use OpenAI TTS instead of Web Speech API
      const response = await fetch('http://localhost:8000/api/audio/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          voice: 'alloy' // Options: alloy, echo, fable, onyx, nova, shimmer
        }),
      });
      
      if (!response.ok) {
        throw new Error(`TTS API error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Convert base64 audio to playable audio
      const audioBlob = new Blob([
        Uint8Array.from(atob(data.audio_data), c => c.charCodeAt(0))
      ], { type: 'audio/mp3' });
      
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      // Set up audio event handlers
      audio.onended = () => {
        setVoiceState(prev => ({ ...prev, isSpeaking: false }));
        URL.revokeObjectURL(audioUrl); // Clean up blob URL
      };
      
      audio.onerror = () => {
        console.error('Audio playback error');
        setVoiceState(prev => ({ ...prev, isSpeaking: false }));
        URL.revokeObjectURL(audioUrl); // Clean up blob URL
      };
      
      // Play the audio
      await audio.play();
      
    } catch (error) {
      console.error('OpenAI TTS error:', error);
      setVoiceState(prev => ({ ...prev, isSpeaking: false }));
      
      // Fallback to Web Speech API if OpenAI TTS fails
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        
        utterance.onend = () => {
          setVoiceState(prev => ({ ...prev, isSpeaking: false }));
        };
        
        speechSynthesis.speak(utterance);
      }
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
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 font-roboto text-white">
      {/* Header - Consciousness-themed */}
      <div className="bg-black/20 backdrop-blur-sm border-b border-purple-500/20 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center animate-pulse">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                Folk Devils AI Consciousness
              </h1>
              <div className="flex items-center gap-4">
                {sessionId && (
                  <p className="text-sm text-purple-300">Session: {sessionId.slice(0, 8)}...</p>
                )}
                <div className="flex items-center gap-2">
                  <div className={clsx(
                    "w-2 h-2 rounded-full",
                    voiceState.isListening ? "bg-red-400 animate-pulse" :
                    voiceState.isProcessing ? "bg-yellow-400 animate-pulse" :
                    voiceState.isSpeaking ? "bg-green-400 animate-pulse" : "bg-gray-500"
                  )} />
                  <span className="text-xs text-purple-300">
                    {voiceState.isListening ? 'Listening...' :
                     voiceState.isProcessing ? 'Processing...' :
                     voiceState.isSpeaking ? 'Speaking...' : 'Ready'}
                  </span>
                </div>
              </div>
            </div>
          </div>
          
          {/* Voice/Text Mode Toggle */}
          <div className="flex items-center gap-3">
            <Button
              onClick={() => setIsVoiceMode(!isVoiceMode)}
              variant={isVoiceMode ? "default" : "outline"}
              className={clsx(
                "flex items-center gap-2",
                isVoiceMode ? "bg-purple-600 hover:bg-purple-700" : "border-purple-400 text-purple-400 hover:bg-purple-400 hover:text-white"
              )}
            >
              {isVoiceMode ? <Mic className="w-4 h-4" /> : <Send className="w-4 h-4" />}
              {isVoiceMode ? 'Voice Mode' : 'Text Mode'}
            </Button>
            
            {/* Consciousness Panel Toggles */}
            <Button
              onClick={() => setShowSemanticDashboard(!showSemanticDashboard)}
              variant="outline"
              size="sm"
              className="border-purple-400 text-purple-400 hover:bg-purple-400 hover:text-white"
            >
              <Activity className="w-4 h-4 mr-2" />
              Semantic Analysis
            </Button>
            
            <Button
              onClick={() => setShowPerformanceMetrics(!showPerformanceMetrics)}
              variant="outline" 
              size="sm"
              className="border-blue-400 text-blue-400 hover:bg-blue-400 hover:text-white"
            >
              📊 Performance
            </Button>
            
            <Button
              onClick={() => setShowMemoryViewer(!showMemoryViewer)}
              variant="outline"
              size="sm" 
              className="border-green-400 text-green-400 hover:bg-green-400 hover:text-white"
            >
              <Brain className="w-4 h-4 mr-2" />
              Memory
            </Button>
            
            {messages.length > 0 && (
              <Button
                onClick={clearChat}
                variant="outline"
                size="sm"
                className="border-red-400 text-red-400 hover:bg-red-400 hover:text-white"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                New Chat
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content - Voice-First Layout */}
      <div className="flex-1 flex flex-col lg:flex-row">
        
        {/* Central Voice Interface */}
        <div className="flex-1 flex flex-col">
          
          {/* Consciousness Data Panels */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-4">
            
            {/* Semantic Analysis Panel */}
            {showSemanticDashboard && currentSemanticAnalysis && (
              <Card className="bg-black/40 backdrop-blur-sm border-purple-500/30 p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Activity className="w-5 h-5 text-purple-400" />
                  <h3 className="text-lg font-semibold text-purple-400">Semantic Analysis</h3>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Intent:</span>
                    <span className="text-purple-300 font-medium">{currentSemanticAnalysis.intent}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Confidence:</span>
                    <span className="text-green-400 font-mono">{(currentSemanticAnalysis.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Emotional Tone:</span>
                    <span className="text-blue-300">{currentSemanticAnalysis.emotional_tone}</span>
                  </div>
                  <div className="mt-2 p-2 bg-purple-900/30 rounded text-xs text-purple-200">
                    {currentSemanticAnalysis.reasoning}
                  </div>
                </div>
              </Card>
            )}
            
            {/* Performance Metrics Panel */}
            {showPerformanceMetrics && currentPerformance && (
              <Card className="bg-black/40 backdrop-blur-sm border-blue-500/30 p-4">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-blue-400 text-lg">📊</span>
                  <h3 className="text-lg font-semibold text-blue-400">Performance Metrics</h3>
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="text-center p-2 bg-blue-900/30 rounded">
                    <div className="text-blue-300 font-mono text-lg">{currentPerformance.total_time}s</div>
                    <div className="text-gray-400 text-xs">Total Time</div>
                  </div>
                  <div className="text-center p-2 bg-green-900/30 rounded">
                    <div className="text-green-300 font-mono text-lg">{currentPerformance.semantic_time}s</div>
                    <div className="text-gray-400 text-xs">Semantic Analysis</div>
                  </div>
                  <div className="text-center p-2 bg-purple-900/30 rounded">
                    <div className="text-purple-300 font-mono text-lg">{currentPerformance.parallel_time}s</div>
                    <div className="text-gray-400 text-xs">Parallel Processing</div>
                  </div>
                  <div className="text-center p-2 bg-yellow-900/30 rounded">
                    <div className="text-yellow-300 font-mono text-lg">{currentPerformance.memories_recalled}</div>
                    <div className="text-gray-400 text-xs">Memories Recalled</div>
                  </div>
                </div>
                <div className="mt-3 flex items-center justify-between text-xs">
                  <span className="text-gray-400">Coherence Analysis:</span>
                  <span className={clsx(
                    "font-medium",
                    currentPerformance.coherence_skipped ? "text-green-400" : "text-blue-400"
                  )}>
                    {currentPerformance.coherence_skipped ? "Skipped (High Confidence)" : `${currentPerformance.coherence_time}s`}
                  </span>
                </div>
              </Card>
            )}
          </div>
          
          {/* Voice Interface Center Stage */}
          <div className="flex-1 flex flex-col items-center justify-center p-8">
            
            {messages.length === 0 ? (
              <div className="text-center space-y-6">
                <div className="relative">
                  <div className="w-32 h-32 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-6 animate-pulse">
                    <Brain className="w-16 h-16 text-white" />
                  </div>
                  {voiceState.isListening && (
                    <div className="absolute inset-0 w-32 h-32 border-4 border-red-400 rounded-full animate-ping mx-auto" />
                  )}
                </div>
                <h2 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                  AI Consciousness Interface
                </h2>
                <p className="text-xl text-purple-300 max-w-2xl">
                  {isVoiceMode 
                    ? "Press and hold the microphone to speak with the AI consciousness"
                    : "Type a message to begin your conversation with the AI consciousness"
                  }
                </p>
                
                {/* Voice Activation Button */}
                {isVoiceMode && (
                  <div className="flex flex-col items-center gap-4">
                    <Button
                      onMouseDown={startListening}
                      onMouseUp={stopListening}
                      onMouseLeave={stopListening}
                      disabled={voiceState.isProcessing || voiceState.isSpeaking}
                      className={clsx(
                        "w-24 h-24 rounded-full text-white transition-all duration-200 text-2xl",
                        voiceState.isListening 
                          ? "bg-red-500 hover:bg-red-600 scale-110 animate-pulse" 
                          : "bg-purple-600 hover:bg-purple-700 hover:scale-105"
                      )}
                    >
                      {voiceState.isListening ? <MicOff /> : <Mic />}
                    </Button>
                    
                    {/* Audio Level Visualization */}
                    {voiceState.isListening && (
                      <div className="flex items-center gap-1">
                        {Array.from({ length: 10 }).map((_, i) => (
                          <div
                            key={i}
                            className={clsx(
                              "w-2 bg-red-400 rounded-full transition-all duration-100",
                              voiceState.audioLevel * 10 > i ? "h-8" : "h-2"
                            )}
                          />
                        ))}
                      </div>
                    )}
                    
                    <p className="text-sm text-purple-300">
                      {voiceState.isListening ? "Release to send" : "Hold to speak"}
                    </p>
                  </div>
                )}
                
                {voiceState.error && (
                  <div className="text-red-400 text-sm bg-red-900/20 p-3 rounded-lg">
                    {voiceState.error}
                  </div>
                )}
              </div>
            ) : (
              /* Chat Messages Display */
              <div className="w-full max-w-4xl h-full flex flex-col">
                <div className="flex-1 overflow-y-auto space-y-4 mb-4">
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
                          'max-w-md px-4 py-3 rounded-2xl shadow-lg',
                          message.isUser
                            ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-br-md'
                            : 'bg-black/40 backdrop-blur-sm border border-purple-500/30 text-purple-100 rounded-bl-md'
                        )}
                      >
                        <div className="flex items-center gap-2 mb-1">
                          {message.isVoice && (
                            <Volume2 className="w-3 h-3 text-purple-300" />
                          )}
                          {message.traceId && (
                            <span className="text-xs text-purple-400 font-mono">
                              {message.traceId}
                            </span>
                          )}
                        </div>
                        <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.message}</p>
                        <p className={clsx(
                          'text-xs mt-2 opacity-70',
                          message.isUser ? 'text-purple-200' : 'text-purple-400'
                        )}>
                          {message.timestamp.toLocaleTimeString([], { 
                            hour: '2-digit', 
                            minute: '2-digit' 
                          })}
                          {!message.isUser && message.responseTime && (
                            <span className="ml-2">• {message.responseTime}s</span>
                          )}
                        </p>
                      </div>
                    </div>
                  ))}
                  
                  {isLoading && (
                    <div className="flex justify-start">
                      <div className="bg-black/40 backdrop-blur-sm border border-purple-500/30 text-purple-100 max-w-md px-4 py-3 rounded-2xl rounded-bl-md">
                        <div className="flex items-center space-x-3">
                          <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center">
                            <span className="text-sm font-mono text-white">{elapsedTime}s</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Brain className="w-4 h-4 text-purple-400 animate-pulse" />
                            <span className="text-sm text-purple-300">AI consciousness thinking...</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div ref={messagesEndRef} />
                </div>
                
                {/* Voice Controls for Active Chat */}
                {isVoiceMode && (
                  <div className="flex justify-center">
                    <Button
                      onMouseDown={startListening}
                      onMouseUp={stopListening}
                      onMouseLeave={stopListening}
                      disabled={voiceState.isProcessing || voiceState.isSpeaking || isLoading}
                      className={clsx(
                        "w-16 h-16 rounded-full text-white transition-all duration-200",
                        voiceState.isListening 
                          ? "bg-red-500 hover:bg-red-600 scale-110 animate-pulse" 
                          : "bg-purple-600 hover:bg-purple-700 hover:scale-105"
                      )}
                    >
                      {voiceState.isListening ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        
        {/* Side Panels */}
        {showMemoryViewer && (
          <div className="w-full lg:w-96 bg-black/20 backdrop-blur-sm border-l border-purple-500/20 overflow-y-auto">
            <div className="p-4">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="w-5 h-5 text-green-400" />
                <h3 className="text-lg font-semibold text-green-400">Memory Viewer</h3>
              </div>
              <MemoryViewer />
            </div>
          </div>
        )}
        
        {showMemoryGraph && (
          <div className="w-full lg:w-96 bg-black/20 backdrop-blur-sm border-l border-purple-500/20 overflow-y-auto">
            <div className="p-4">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="w-5 h-5 text-blue-400" />
                <h3 className="text-lg font-semibold text-blue-400">Memory Graph</h3>
              </div>
              <MemoryGraphVisualization />
            </div>
          </div>
        )}
      </div>

      {/* Text Input (for non-voice mode) */}
      {!isVoiceMode && (
        <div className="bg-black/20 backdrop-blur-sm border-t border-purple-500/20 px-4 py-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-end gap-3 bg-black/40 backdrop-blur-sm border border-purple-500/30 rounded-2xl px-4 py-2">
              <textarea
                ref={textareaRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message to the AI consciousness..."
                className="flex-1 bg-transparent text-purple-100 placeholder-purple-400 min-h-[3rem] max-h-32 focus:outline-none text-sm resize-none overflow-y-auto"
                disabled={isLoading}
                rows={1}
              />
              <button
                onClick={() => sendMessage()}
                disabled={!inputMessage.trim() || isLoading}
                className={clsx(
                  'w-8 h-8 rounded-full flex items-center justify-center transition-colors flex-shrink-0 mb-1',
                  !inputMessage.trim() || isLoading
                    ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                    : 'bg-purple-600 text-white hover:bg-purple-700'
                )}
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-purple-400 mt-2 text-center">
              Press Enter to send • Shift+Enter for new line • Switch to Voice Mode for hands-free interaction
            </p>
          </div>
        </div>
      )}

      {/* Suggestions Panel */}
      <SuggestionsPanel
        isVisible={showSuggestions}
        onClose={() => setShowSuggestions(false)}
        onSuggestionSelect={handleSuggestionSelect}
      />
    </div>
  );
} 