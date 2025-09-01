'use client';

import { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, RefreshCw, Brain, Volume2 } from 'lucide-react';
import { clsx } from 'clsx';
// Removed dashboard components for simplified interface
import { Button } from '../components/ui/button';
// Removed Card import as we no longer use dashboard cards

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
  const [isVoiceMode] = useState(true); // Always voice-only mode
  const [lastProcessedTranscript, setLastProcessedTranscript] = useState<string>('');
  
  // Refs
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



  const clearChat = () => {
    setMessages([]);
    setSessionId(null);
    clearCache();
  };



  return (
    <div className="fixed inset-0 bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 font-roboto text-white overflow-hidden">{/* Pure Voice Interface - Full Screen Background */}

      {/* Pure Voice Interface - Centered */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        
        {/* Status Display */}
        <div className="text-center mb-12">
          {voiceState.isProcessing ? (
            <div className="space-y-4">
              <div className="w-20 h-20 bg-yellow-500/20 rounded-full flex items-center justify-center mx-auto animate-pulse">
                <Brain className="w-10 h-10 text-yellow-400 animate-pulse" />
              </div>
              <p className="text-xl text-yellow-300">Processing...</p>
            </div>
          ) : voiceState.isSpeaking ? (
            <div className="space-y-4">
              <div className="w-20 h-20 bg-green-500/20 rounded-full flex items-center justify-center mx-auto animate-pulse">
                <Volume2 className="w-10 h-10 text-green-400 animate-pulse" />
              </div>
              <p className="text-xl text-green-300">Speaking...</p>
            </div>
          ) : voiceState.isListening ? (
            <div className="space-y-4">
              <div className="w-20 h-20 bg-red-500/20 rounded-full flex items-center justify-center mx-auto animate-pulse">
                <Mic className="w-10 h-10 text-red-400 animate-pulse" />
              </div>
              <p className="text-xl text-red-300">Listening...</p>
              {/* Audio Level Visualization */}
              <div className="flex items-center justify-center gap-1">
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
            </div>
          ) : (
            <div className="space-y-4">
              <div className="w-20 h-20 bg-purple-500/20 rounded-full flex items-center justify-center mx-auto">
                <Brain className="w-10 h-10 text-purple-400" />
              </div>
              <p className="text-xl text-purple-300">Ready</p>
            </div>
          )}
        </div>

        {/* Voice Control Button */}
        <div className="flex flex-col items-center gap-6">
          <Button
            onMouseDown={startListening}
            onMouseUp={stopListening}
            onMouseLeave={stopListening}
            disabled={voiceState.isProcessing || voiceState.isSpeaking || isLoading}
            className={clsx(
              "w-32 h-32 rounded-full text-white transition-all duration-200 text-3xl shadow-2xl",
              voiceState.isListening 
                ? "bg-red-500 hover:bg-red-600 scale-110 animate-pulse shadow-red-500/50" 
                : "bg-purple-600 hover:bg-purple-700 hover:scale-105 shadow-purple-500/50"
            )}
          >
            {voiceState.isListening ? <MicOff className="w-12 h-12" /> : <Mic className="w-12 h-12" />}
          </Button>
          
          <p className="text-lg text-purple-300">
            {voiceState.isListening ? "Release to send" : "Hold to speak"}
          </p>
        </div>

        {/* Error Display */}
        {voiceState.error && (
          <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2">
            <div className="text-red-400 text-sm bg-red-900/20 p-3 rounded-lg backdrop-blur-sm">
              {voiceState.error}
            </div>
          </div>
        )}

        {/* Clear Chat Button - Hidden but functional */}
        {messages.length > 0 && (
          <button
            onClick={clearChat}
            className="absolute top-4 right-4 text-purple-400 hover:text-white opacity-50 hover:opacity-100 transition-opacity"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        )}
      </div>




    </div>
  );
} 