'use client';

import { useState, useEffect, useRef } from 'react';
import { Mic, MicOff } from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from '../../components/ui/button';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useControls, Leva } from 'leva';
import { EffectComposer, Bloom, DepthOfField } from '@react-three/postprocessing';

interface ChatMessage {
  id: string;
  message: string;
  isUser: boolean;
  timestamp: Date;
  responseTime?: number;
  isVoice?: boolean;
  traceId?: string;
}

interface ChatResponse {
  response: string;
  session_id: string;
  command_used?: string;
  reasoning?: string;
  memory_context?: string;
  voice_compliance_score?: number;
  timestamp: number;
  cache_version: string;
  trace_id?: string;
}

interface VoiceState {
  isListening: boolean;
  isProcessing: boolean;
  isSpeaking: boolean;
  audioLevel: number;
  error?: string;
}

// Voice-reactive undulating vertex shader
const vertexShader = `
  uniform float u_time;
  uniform float u_audioLevel;
  
  varying vec3 vNormal;
  varying vec3 vPosition;
  
  void main() {
    vec3 pos = position;
    
    // More visible base undulation when not speaking
    float wave1 = sin(pos.x * 2.0 + u_time) * 0.12;
    float wave2 = sin(pos.y * 1.5 + u_time * 0.7) * 0.10;
    float wave3 = sin(pos.z * 1.8 + u_time * 0.5) * 0.08;
    
    // Voice-reactive waves - slower time, slightly lower amplitude (smoother)
    float audioWave1 = sin(pos.x * 3.2 + u_time * 2.2) * u_audioLevel * 0.50;
    float audioWave2 = sin(pos.y * 2.9 + u_time * 1.8) * u_audioLevel * 0.44;
    float audioWave3 = sin(pos.z * 3.1 + u_time * 2.0) * u_audioLevel * 0.40;
    
    // Speech waves - reduce high-frequency jitter
    float speechWave1 = sin(pos.x * 6.0 + u_time * 4.0) * u_audioLevel * 0.20;
    float speechWave2 = sin(pos.y * 5.5 + u_time * 3.8) * u_audioLevel * 0.18;
    float speechWave3 = sin(pos.z * 5.0 + u_time * 3.2) * u_audioLevel * 0.16;
    
    // Combine base + voice reactive + speech waves
    float totalDeformation = wave1 + wave2 + wave3 + audioWave1 + audioWave2 + audioWave3 + speechWave1 + speechWave2 + speechWave3;
    
    // Apply undulation along the normal direction
    vec3 undulatedPos = pos + normalize(pos) * totalDeformation;
    
    // Calculate the new normal after deformation
    vec3 newNormal = normalize(undulatedPos);
    vNormal = normalMatrix * newNormal;
    vPosition = undulatedPos;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(undulatedPos, 1.0);
  }
`;

// Clean glass blob with internal vein structure
function VoiceBlob({ audioLevel, gradientColors, blendMode }: { audioLevel: number, gradientColors: any, blendMode: string }) {
  const glassMeshRef = useRef<THREE.Mesh>(null);
  const glassMaterialRef = useRef<THREE.ShaderMaterial>(null);
  
  // Create fragment shader with dynamic colors
  const createFragmentShader = (deepColor: string, highColor: string) => {
    const deep = new THREE.Color(deepColor);
    const high = new THREE.Color(highColor);
    
    return `
      varying vec3 vNormal;
      varying vec3 vPosition;
      
      void main() {
        // Normalize the normal vector
        vec3 normal = normalize(vNormal);
        
        // Gradient based on distance from center - adjusted for smaller sphere
        float depth = length(vPosition);
        
        // Adjusted mapping for 0.8 radius sphere with undulation
        float normalizedDepth = (depth - 0.7) / 0.4; // Better range for smaller sphere
        normalizedDepth = clamp(normalizedDepth, 0.0, 1.0);
        
        // Dynamic colors from controls
        vec3 deepColor = vec3(${deep.r.toFixed(3)}, ${deep.g.toFixed(3)}, ${deep.b.toFixed(3)});
        vec3 highColor = vec3(${high.r.toFixed(3)}, ${high.g.toFixed(3)}, ${high.b.toFixed(3)});
        
        // Mix colors - should be obvious gradient
        vec3 gradientColor = mix(deepColor, highColor, normalizedDepth);
        
        // Subtle glass effects that don't wash out the gradient
        vec3 viewDirection = normalize(-vPosition);
        float rim = 1.0 - dot(normal, viewDirection);
        rim = pow(rim, 3.0) * 0.15; // Much more subtle rim effect
        
        // Very gentle directional lighting
        vec3 lightDirection = normalize(vec3(1.0, 1.0, 0.5));
        float lightIntensity = dot(normal, lightDirection) * 0.1 + 0.9; // Subtle lighting
        
        // Combine: mostly gradient with subtle glass effects
        vec3 finalColor = gradientColor * lightIntensity + gradientColor * rim;
        
        // Add emissive glow for bloom effect - make it glow!
        vec3 emissiveGlow = gradientColor * 0.4; // Make the colors glow
        finalColor += emissiveGlow;
        
        gl_FragColor = vec4(finalColor, 0.85);
      }
    `;
  };
  
  // Get blend mode for Three.js - using only stable blend modes
  const getBlendMode = (mode: string) => {
    switch(mode) {
      case 'multiply': return THREE.MultiplyBlending;
      case 'screen': return THREE.AdditiveBlending;
      case 'add': return THREE.AdditiveBlending;
      case 'overlay': return THREE.AdditiveBlending; // Use additive instead
      case 'difference': return THREE.AdditiveBlending; // Use additive instead of problematic subtractive
      default: return THREE.NormalBlending;
    }
  };

  // Check if blend mode needs premultiplied alpha
  const needsPremultipliedAlpha = (mode: string) => {
    return mode === 'multiply';
  };

  // Voice-reactive material with dynamic colors
  const voiceMaterial = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader: createFragmentShader(gradientColors.deepColor, gradientColors.highColor),
    uniforms: {
      u_time: { value: 0 },
      u_audioLevel: { value: 0 }
    },
    transparent: true,
    blending: getBlendMode(blendMode),
    premultipliedAlpha: needsPremultipliedAlpha(blendMode)
  });
  
  // Update material when colors or blend mode change
  useEffect(() => {
    if (materialRef.current) {
      materialRef.current.fragmentShader = createFragmentShader(gradientColors.deepColor, gradientColors.highColor);
      materialRef.current.blending = getBlendMode(blendMode);
      materialRef.current.premultipliedAlpha = needsPremultipliedAlpha(blendMode);
      materialRef.current.needsUpdate = true;
    }
  }, [gradientColors, blendMode]);
  
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  
  useFrame((state) => {
    if (!materialRef.current) return;
    materialRef.current.uniforms.u_time.value = state.clock.elapsedTime;
    materialRef.current.uniforms.u_audioLevel.value = audioLevel;
  });

  return (
    <group>
      {/* SIMPLE SPHERE - NO GEOMETRY ARTIFACTS */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.8, 128, 64]} />
        <primitive 
          ref={materialRef}
          object={voiceMaterial} 
          attach="material"
        />
      </mesh>
    </group>
  );
}

// Simple hemisphere lighting for even, beautiful illumination
function Lights() {
  return (
    <>
      {/* Hemisphere light for smooth, even lighting */}
      <hemisphereLight 
        args={["#ffffff", "#4a0e4e", 1.2]}
      />
      
      {/* Single soft directional light */}
      <directionalLight 
        position={[5, 5, 5]} 
        intensity={0.3} 
        color="#ffffff" 
      />
    </>
  );
}

export default function VoiceAnimationIntegratedTest() {
  // State for controlling Leva controls visibility
  const [levaVisible, setLevaVisible] = useState(true);

  // Leva controls for real-time color adjustment
  const { backgroundTop, backgroundBottom, gradientDeep, gradientHigh, bloomIntensity, bloomRadius, blurIntensity, blendMode, buttonOffColor, buttonOnColor } = useControls({
    backgroundTop: { value: '#8f2eff', label: 'Background Top' },
    backgroundBottom: { value: '#4f004d', label: 'Background Bottom' },
    gradientDeep: { value: '#d89e9e', label: 'Gradient Deep' },
    gradientHigh: { value: '#5400b5', label: 'Gradient High' },
    bloomIntensity: { value: 2.5, min: 0, max: 5, step: 0.1, label: 'Bloom Intensity' },
    bloomRadius: { value: 0.4, min: 0, max: 2, step: 0.1, label: 'Bloom Radius' },
    blurIntensity: { value: .15, min: 0, max: 5, step: 0.05, label: 'Blur Intensity' },
    blendMode: { 
      value: 'normal', 
      options: ['normal', 'multiply', 'screen', 'overlay', 'add', 'difference'],
      label: 'Blend Mode' 
    },
    buttonOffColor: { value: '#9333ea', label: 'Button Off Color' },
    buttonOnColor: { value: '#d600ff', label: 'Button On Color' }
  });

  const [voiceState, setVoiceState] = useState<VoiceState>({
    isListening: false,
    isProcessing: false,
    isSpeaking: false,
    audioLevel: 0
  });

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [lastProcessedTranscript, setLastProcessedTranscript] = useState<string>('');

  // Audio analysis refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioWebSocketRef = useRef<WebSocket | null>(null);
  const recognitionRef = useRef<any>(null);
  const websocketTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  // Animation analysis refs (separate from playback)
  const ttsAnalysisContextRef = useRef<AudioContext | null>(null);
  const ttsAnalyserRef = useRef<AnalyserNode | null>(null);
  const ttsAnimationFrameRef = useRef<number | null>(null);
  // Gate for button press - only process transcriptions when button is actively pressed
  const isButtonPressedRef = useRef<boolean>(false);
  
  // Temporal coherence - smooth audio transitions
  const smoothedAudioRef = useRef<number>(0);
  const audioVelocityRef = useRef<number>(0);

  const startListening = async () => {
    try {
      console.log('🎤 Starting listening...');
      setVoiceState(prev => ({ ...prev, isListening: true, error: undefined }));
      setLastProcessedTranscript('');
      // Set button pressed flag to true
      isButtonPressedRef.current = true;
      console.log('🎤 Button pressed flag set to true');

      // Prefer Web Speech API (matches main page)
      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';
        recognitionRef.current = recognition;

        recognition.onresult = (event: any) => {
          const latestResult = event.results[event.results.length - 1][0];
          console.log('🎤 Speech recognition result received:', {
            transcript: latestResult.transcript,
            isFinal: event.results[event.results.length - 1].isFinal,
            buttonPressed: isButtonPressedRef.current,
            confidence: latestResult.confidence
          });

          // CRITICAL: Only process transcriptions when button is actively pressed
          if (!isButtonPressedRef.current) {
            console.log('🎤 Ignoring transcription - button not pressed:', latestResult.transcript);
            return;
          }
          
          let finalTranscript = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const t = event.results[i][0].transcript;
            if (event.results[i].isFinal) finalTranscript += t;
          }
          if (finalTranscript.trim()) {
            console.log('🎤 Processing transcription - button pressed:', finalTranscript.trim());
            processVoiceTranscript(finalTranscript.trim(), event.results[event.results.length - 1][0].confidence);
          }
        };
        recognition.onerror = () => setVoiceState(prev => ({ ...prev, isListening: false }));
        recognition.onend = () => setVoiceState(prev => ({ ...prev, isListening: false }));
        recognition.start();

        // Ensure WS is ready to receive transcribed text (matches main page behavior)
        setupAudioWebSocket();
      }

      // Always set up audio visualization for the blob
      await setupAudioVisualization();
    } catch (error) {
      console.error('Error starting voice recording:', error);
      setVoiceState(prev => ({ ...prev, isListening: false, error: 'Microphone access denied or not available' }));
    }
  };

  const setupAudioVisualization = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      const microphone = audioContext.createMediaStreamSource(stream);
      microphone.connect(analyser);
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.3;
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      const updateAudioLevel = () => {
        if (analyserRef.current) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          
          // RMS calculation for better audio response (matches reference page)
          const sum = dataArray.reduce((acc, val) => acc + val * val, 0);
          const rms = Math.sqrt(sum / dataArray.length);
          
          // Apply exponential scaling for more dramatic response
          let normalizedLevel = Math.min(rms / 64, 1); // Normalize to 0-1
          
          // Lower threshold for more sensitivity to quick sounds
          if (normalizedLevel < 0.02) {
            normalizedLevel = 0;
          } else {
            // More aggressive boost for responsiveness
            normalizedLevel = Math.pow(normalizedLevel, 0.5); // More sensitive power curve
          }
          
          // Much more responsive temporal coherence
          const currentSmoothed = smoothedAudioRef.current;
          const targetLevel = normalizedLevel;
          
          // Calculate velocity-based smoothing - more responsive
          const difference = targetLevel - currentSmoothed;
          const smoothingFactor = 0.4; // Much more responsive
          const velocityDamping = 0.6; // Less damping for quicker response
          
          // Update velocity with acceleration towards target
          audioVelocityRef.current = audioVelocityRef.current * velocityDamping + difference * smoothingFactor;
          
          // Update smoothed value with velocity
          smoothedAudioRef.current = currentSmoothed + audioVelocityRef.current;
          
          // Clamp to valid range
          smoothedAudioRef.current = Math.max(0, Math.min(1, smoothedAudioRef.current));
          
          setVoiceState(prev => ({ ...prev, audioLevel: smoothedAudioRef.current }));
          animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
        }
      };
      updateAudioLevel();
    } catch (e) {
      console.warn('Audio visualization setup failed:', e);
    }
  };

  const setupAudioWebSocket = () => {
    try {
      // Close existing WebSocket if it exists
      if (audioWebSocketRef.current) {
        audioWebSocketRef.current.close();
        audioWebSocketRef.current = null;
      }

      const conversationId = sessionId || `conv_${Date.now()}`;
      const wsUrl = `ws://localhost:8000/ws/audio/${conversationId}`;
      console.log('🎤 Connecting to WebSocket:', wsUrl);
      const ws = new WebSocket(wsUrl);
      audioWebSocketRef.current = ws;

      ws.onopen = () => {
        console.log('🎤 Audio WebSocket connected successfully');
        // Clear any previous connection errors
        setVoiceState(prev => ({ ...prev, error: undefined }));
        ws.send(JSON.stringify({ type: 'start_voice_session', timestamp: new Date().toISOString() }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('🎤 Audio WebSocket message:', data.type, data);

          if (data.type === "consciousness_response") {
            // Handle consciousness response from voice input
            const responseData = data.data;
            console.log('🎤 Received consciousness response:', responseData);
            
            // Clear the WebSocket timeout if it exists
            if (websocketTimeoutRef.current) {
              clearTimeout(websocketTimeoutRef.current);
              websocketTimeoutRef.current = null;
            }
            
            // Update session ID
            if (responseData.session_id) {
              setSessionId(responseData.session_id);
            }

            // Speak the response directly
            if (responseData.response) {
              speakResponse(responseData.response);
            }
            
            // Mark voice processing as complete
            setVoiceState(prev => ({ ...prev, isProcessing: false, error: undefined }));
          }
        } catch (error) {
          console.error('🎤 Error parsing audio WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('🎤 Audio WebSocket error:', error);
        setVoiceState(prev => ({ 
          ...prev, 
          error: 'Voice connection failed' 
        }));
      };

      ws.onclose = (event) => {
        console.log('🎤 Audio WebSocket disconnected. Code:', event.code, 'Reason:', event.reason);
        // Don't set error if it was a clean close
        if (event.code !== 1000) {
          setVoiceState(prev => ({ 
            ...prev, 
            error: 'Voice connection lost' 
          }));
        }
      };

    } catch (error) {
      console.error('🎤 Failed to connect audio WebSocket:', error);
      setVoiceState(prev => ({ 
        ...prev, 
        error: 'Failed to establish voice connection' 
      }));
    }
  };

  // Mirror main page: send transcript over WS if available; fallback to HTTP
  const processVoiceTranscript = async (transcript: string, confidence: number) => {
    try {
      console.log('🎤 processVoiceTranscript called:', { transcript, confidence });
      
      if (!transcript.trim()) {
        console.log('🎤 Empty transcript, skipping');
        return;
      }
      if (transcript === lastProcessedTranscript) {
        console.log('🎤 Duplicate transcript, skipping');
        return;
      }
      
      setLastProcessedTranscript(transcript);
      setVoiceState(prev => ({ ...prev, isProcessing: true }));

      // Preferred path: send over WebSocket
      if (audioWebSocketRef.current && audioWebSocketRef.current.readyState === WebSocket.OPEN) {
        console.log('🎤 Sending transcript via WebSocket:', transcript);
        audioWebSocketRef.current.send(JSON.stringify({
          type: 'transcribed_text',
          text: transcript,
          confidence,
          timestamp: new Date().toISOString()
        }));
        
        // Set a timeout to fall back to HTTP if WebSocket doesn't respond
        const websocketTimeout = setTimeout(() => {
          console.log('🎤 WebSocket response timeout, falling back to HTTP');
          setVoiceState(prev => ({ ...prev, isProcessing: false, error: 'WebSocket timeout' }));
        }, 10000); // 10 second timeout
        
        // Store timeout in ref for cleanup
        websocketTimeoutRef.current = websocketTimeout;
        return; // WS handler will trigger TTS and clear processing
      }

      console.log('🎤 WebSocket not available, using HTTP fallback. WS state:', 
        audioWebSocketRef.current ? audioWebSocketRef.current.readyState : 'null');

      // Fallback: HTTP streaming endpoint (same as main page)
      const response = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: transcript, speaker: 'andrew' })
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      console.log('🎤 HTTP response received:', data);
      if (data?.response) await speakResponse(data.response);
      setVoiceState(prev => ({ ...prev, isProcessing: false }));
    } catch (err) {
      console.error('processVoiceTranscript error:', err);
      setVoiceState(prev => ({ ...prev, isProcessing: false, error: 'Failed to process transcript' }));
    }
  };

  // processTranscript removed - WebSocket handles everything directly

  const speakResponse = async (text: string) => {
    try {
      setVoiceState(prev => ({ ...prev, isSpeaking: true, isProcessing: false }));
      
      // Use the same TTS endpoint as main page
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
      
             // Monitor audio playback for visualization - ANALYSIS ONLY, no routing
       audio.addEventListener('play', async () => {
         console.log('🔊 TTS playback started');
         try {
           // Create separate analysis context that doesn't interfere with playback
           const analysisCtx = new AudioContext();
           const source = analysisCtx.createMediaElementSource(audio);
           const analyser = analysisCtx.createAnalyser();
           
           // CRITICAL: Connect source to destination for normal playback
           source.connect(analysisCtx.destination);
           // ALSO connect to analyser for animation data
           source.connect(analyser);
           
           analyser.fftSize = 512;
           analyser.smoothingTimeConstant = 0.3; // Much less smoothing for sharp word transitions
           
           ttsAnalysisContextRef.current = analysisCtx;
           ttsAnalyserRef.current = analyser;
           
           // More responsive animation loop for AI response
           const dataArray = new Uint8Array(analyser.frequencyBinCount);
           let smoothLevel = 0;
           const smoothingFactor = 0.7; // Much more aggressive for sharp word pulsing
           
           const animateResponse = () => {
             if (!ttsAnalyserRef.current || audio.paused || audio.ended) return;
             
             ttsAnalyserRef.current.getByteFrequencyData(dataArray);
             
             // Use RMS like recording for consistency
             const sum = dataArray.reduce((acc, val) => acc + val * val, 0);
             const rms = Math.sqrt(sum / dataArray.length);
             let normalizedLevel = Math.min(rms / 64, 1);
             
             // Apply noise floor and power curve for sharp word response
             if (normalizedLevel < 0.01) {
               normalizedLevel = 0;
             } else {
               normalizedLevel = Math.pow(normalizedLevel, 0.3); // Very aggressive for sharp word pulsing
             }
             
             // More responsive smoothing
             smoothLevel = smoothLevel * (1 - smoothingFactor) + normalizedLevel * smoothingFactor;
             
             setVoiceState(prev => ({ ...prev, audioLevel: smoothLevel }));
             ttsAnimationFrameRef.current = requestAnimationFrame(animateResponse);
           };
           animateResponse();
           
         } catch (err) {
           console.warn('TTS analysis failed, using simulation:', err);
           simulateAudioLevelDuringPlayback(audio);
         }
       });
      
      // Set up audio event handlers
      audio.onended = () => {
        console.log('🔊 TTS playback ended');
        setVoiceState(prev => ({ ...prev, isSpeaking: false, audioLevel: 0 }));
        URL.revokeObjectURL(audioUrl); // Clean up blob URL
        
        // Clean up analysis resources
        if (ttsAnimationFrameRef.current) {
          cancelAnimationFrame(ttsAnimationFrameRef.current);
          ttsAnimationFrameRef.current = null;
        }
        if (ttsAnalysisContextRef.current) {
          ttsAnalysisContextRef.current.close();
          ttsAnalysisContextRef.current = null;
          ttsAnalyserRef.current = null;
        }
      };
      
      audio.onerror = () => {
        console.error('🔊 TTS playback error');
        setVoiceState(prev => ({ ...prev, isSpeaking: false, audioLevel: 0 }));
        URL.revokeObjectURL(audioUrl); // Clean up blob URL
        
        // Clean up analysis resources
        if (ttsAnimationFrameRef.current) {
          cancelAnimationFrame(ttsAnimationFrameRef.current);
          ttsAnimationFrameRef.current = null;
        }
        if (ttsAnalysisContextRef.current) {
          ttsAnalysisContextRef.current.close();
          ttsAnalysisContextRef.current = null;
          ttsAnalyserRef.current = null;
        }
      };
      
      // Play the audio
      await audio.play();
      
    } catch (error) {
      console.error('🔊 OpenAI TTS error:', error);
      setVoiceState(prev => ({ 
        ...prev, 
        isSpeaking: false, 
        isProcessing: false,
        error: 'Failed to play response'
      }));
    }
  };

  const simulateAudioLevelDuringPlayback = (audio: HTMLAudioElement) => {
    const updatePlaybackLevel = () => {
      if (audio.paused || audio.ended) {
        setVoiceState(prev => ({ ...prev, audioLevel: 0 }));
        if (ttsAnimationFrameRef.current) {
          cancelAnimationFrame(ttsAnimationFrameRef.current);
          ttsAnimationFrameRef.current = null;
        }
        return;
      }

      // Simulate audio level based on playback (you could use Web Audio API for real analysis)
      const baseLevel = 0.3 + Math.random() * 0.4; // Random between 0.3-0.7
      const variation = Math.sin(Date.now() * 0.01) * 0.2; // Sine wave variation
      const simulatedLevel = Math.max(0, Math.min(1, baseLevel + variation));

      setVoiceState(prev => ({ ...prev, audioLevel: simulatedLevel }));
      
      // CRITICAL: Store the animation frame ID for cleanup
      ttsAnimationFrameRef.current = requestAnimationFrame(updatePlaybackLevel);
    };
    
    updatePlaybackLevel();
  };

  const stopListening = () => {
    console.log('Stopping listening...'); // Debug log
    
    // Set to processing state when stopping
    setVoiceState(prev => ({ ...prev, isListening: false, isProcessing: true, audioLevel: 0 }));
    
    // CRITICAL: Stop speech recognition first
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    
    // CRITICAL: Clear button pressed flag with delay to allow for speech recognition processing
    setTimeout(() => {
      isButtonPressedRef.current = false;
      console.log('🎤 Button press flag cleared after delay');
    }, 1000); // 1 second delay to allow for speech recognition to process
    
    // Cancel ALL animation frames
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (ttsAnimationFrameRef.current) {
      cancelAnimationFrame(ttsAnimationFrameRef.current);
      ttsAnimationFrameRef.current = null;
    }
    
    // Clear WebSocket timeout if it exists
    if (websocketTimeoutRef.current) {
      clearTimeout(websocketTimeoutRef.current);
      websocketTimeoutRef.current = null;
    }
    
    // Stop media recorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }
    
    // DON'T close WebSocket immediately - keep it open for responses
    // The WebSocket will be closed when a new session starts or component unmounts
    
    // Clean up audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    // Stop media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    // Clear refs
    analyserRef.current = null;
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening();
      
      // Clean up WebSocket on unmount
      if (audioWebSocketRef.current) {
        console.log('🎤 Closing WebSocket on unmount');
        audioWebSocketRef.current.close();
        audioWebSocketRef.current = null;
      }
    };
  }, []);

  // Keyboard event listener for toggling Leva controls
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === '1') {
        setLevaVisible(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, []);

  // Get status text based on voice state
  const getStatusText = () => {
    if (voiceState.isListening) return "Listening...";
    if (voiceState.isProcessing) return "Processing...";
    if (voiceState.isSpeaking) return "Speaking...";
    return "Ready";
  };

  // Get status color based on voice state
  const getStatusColor = () => {
    if (voiceState.isListening) return buttonOnColor;
    if (voiceState.isProcessing) return '#fbbf24'; // Yellow for processing
    if (voiceState.isSpeaking) return '#10b981'; // Green for speaking
    return buttonOffColor;
  };

  // Get button color based on voice state
  const getButtonColor = () => {
    if (voiceState.isListening) return buttonOnColor;
    if (voiceState.isProcessing) return '#fbbf24'; // Yellow for processing
    if (voiceState.isSpeaking) return '#10b981'; // Green for speaking
    return buttonOffColor;
  };

  return (
    <div className="fixed inset-0 font-roboto text-white overflow-hidden" style={{
      background: `linear-gradient(to bottom, ${backgroundTop}, ${backgroundBottom})`
    }}>
      {/* Voice Animation Interface */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        
        {/* 3D Voice Blob Animation - Full Viewport Background */}
        <div className="absolute inset-0 w-full h-full z-0">
          <Canvas camera={{ position: [0, 0, 12], fov: 45 }}>
            <Lights />
            <VoiceBlob audioLevel={voiceState.audioLevel} gradientColors={{deepColor: gradientDeep, highColor: gradientHigh}} blendMode={blendMode} />
            <EffectComposer>
              <Bloom 
                intensity={bloomIntensity}
                radius={bloomRadius}
                luminanceThreshold={0.1}
                luminanceSmoothing={0.9}
              />
              <DepthOfField 
                focusDistance={0}
                focalLength={0.02}
                bokehScale={blurIntensity * 100}
                height={480}
              />
            </EffectComposer>
          </Canvas>
        </div>

        {/* Voice Control Button - Bottom positioned */}
        <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 z-10 flex flex-col items-center gap-3">
          <Button
            onMouseDown={startListening}
            onMouseUp={stopListening}
            onMouseLeave={stopListening}
            disabled={voiceState.isProcessing || voiceState.isSpeaking}
            className={clsx(
              "w-16 h-16 rounded-full text-white transition-all duration-200 text-lg shadow-2xl flex items-center justify-center p-0",
              voiceState.isListening 
                ? "scale-110 animate-pulse" 
                : "hover:scale-105",
              (voiceState.isProcessing || voiceState.isSpeaking) && "opacity-75 cursor-not-allowed"
            )}
            style={{ 
              borderRadius: '50%',
              backgroundColor: getButtonColor(),
              boxShadow: `0 25px 50px -12px ${getButtonColor()}50`
            }}
          >
            {voiceState.isListening ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
          </Button>
          
          {/* Status underneath button */}
          <p className="text-sm" style={{ color: getStatusColor() }}>
            {getStatusText()}
          </p>
        </div>

        {/* Error Display */}
        {voiceState.error && (
          <div className="absolute bottom-32 left-1/2 transform -translate-x-1/2 z-10">
            <div className="text-red-400 text-sm bg-red-900/20 p-3 rounded-lg backdrop-blur-sm">
              {voiceState.error}
            </div>
          </div>
        )}
      </div>

      {/* Leva Controls - Toggle with '1' key */}
      <Leva hidden={!levaVisible} />
    </div>
  );
}