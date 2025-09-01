'use client';

import { useState, useEffect, useRef } from 'react';
import { Mic, MicOff } from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from '../../components/ui/button';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useControls } from 'leva';
import { EffectComposer, Bloom, DepthOfField } from '@react-three/postprocessing';

interface VoiceState {
  isListening: boolean;
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
    
    // Voice-reactive waves - reduced by 20%
    float audioWave1 = sin(pos.x * 4.0 + u_time * 3.0) * u_audioLevel * 0.64;
    float audioWave2 = sin(pos.y * 3.5 + u_time * 2.5) * u_audioLevel * 0.56;
    float audioWave3 = sin(pos.z * 3.8 + u_time * 2.8) * u_audioLevel * 0.48;
    
    // Speech waves - reduced by 20%
    float speechWave1 = sin(pos.x * 8.0 + u_time * 6.0) * u_audioLevel * 0.32;
    float speechWave2 = sin(pos.y * 7.0 + u_time * 5.5) * u_audioLevel * 0.28;
    float speechWave3 = sin(pos.z * 6.5 + u_time * 4.8) * u_audioLevel * 0.24;
    
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

// Transparent glass fragment shader with depth gradient
const fragmentShader = `
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
    
    // EXTREME colors for debugging
    vec3 deepColor = vec3(1.0, 0.0, 0.0);    // PURE RED
    vec3 highColor = vec3(0.0, 0.0, 1.0);    // PURE BLUE
    
    // Mix colors - should be obvious red to blue gradient
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
    
    gl_FragColor = vec4(finalColor, 0.85);
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
      
      {/* Wireframe hidden for clean glass appearance */}
      {/* 
      <mesh ref={wireframeMeshRef} position={[0, 0, 0]} scale={1.001}>
        <sphereGeometry args={[2.5, 64, 64]} />
        <primitive 
          ref={wireframeMaterialRef}
          object={wireframeMaterial} 
          attach="material"
        />
      </mesh>
      */}
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

export default function VoiceAnimationTest() {
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
    audioLevel: 0
  });

  // Audio analysis refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  
  // Temporal coherence - smooth audio transitions
  const smoothedAudioRef = useRef<number>(0);
  const audioVelocityRef = useRef<number>(0);

  const startListening = async () => {
    try {
      console.log('Starting listening...'); // Debug log
      setVoiceState(prev => ({ ...prev, isListening: true, error: undefined }));

      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('Microphone access granted'); // Debug log
      mediaStreamRef.current = stream;
      
      // Set up audio analysis for real-time visualization
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      const microphone = audioContext.createMediaStreamSource(stream);
      
      microphone.connect(analyser);
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.3;

      console.log('Audio context setup complete'); // Debug log
      console.log('FFT size:', analyser.fftSize, 'Frequency bin count:', analyser.frequencyBinCount);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      // Start audio level monitoring for visualization with temporal coherence
      const updateAudioLevel = () => {
        if (analyserRef.current) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          
          // Calculate RMS (Root Mean Square) for better audio detection
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
          
          console.log('Audio level:', smoothedAudioRef.current.toFixed(3)); // Debug log
          setVoiceState(prev => ({ ...prev, audioLevel: smoothedAudioRef.current }));
          animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
        }
      };
      updateAudioLevel();

    } catch (error) {
      console.error('Error starting voice recording:', error);
      setVoiceState(prev => ({ 
        ...prev, 
        isListening: false, 
        error: 'Microphone access denied or not available' 
      }));
    }
  };

  const stopListening = () => {
    console.log('Stopping listening...'); // Debug log
    
    // Cancel animation frame first
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
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
    
    setVoiceState(prev => ({ ...prev, isListening: false, audioLevel: 0 }));
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening();
    };
  }, []);

  return (
    <div className="fixed inset-0 font-roboto text-white overflow-hidden" style={{
      background: `linear-gradient(to bottom, ${backgroundTop}, ${backgroundBottom})`
    }}>
      {/* Voice Animation Test Interface */}
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
            className={clsx(
              "w-16 h-16 rounded-full text-white transition-all duration-200 text-lg shadow-2xl flex items-center justify-center p-0",
              voiceState.isListening 
                ? "scale-110 animate-pulse" 
                : "hover:scale-105"
            )}
            style={{ 
              borderRadius: '50%',
              backgroundColor: voiceState.isListening ? buttonOnColor : buttonOffColor,
              boxShadow: voiceState.isListening 
                ? `0 25px 50px -12px ${buttonOnColor}50` 
                : `0 25px 50px -12px ${buttonOffColor}50`
            }}
          >
            {voiceState.isListening ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
          </Button>
          
          {/* Status underneath button */}
          {voiceState.isListening ? (
            <p className="text-sm" style={{ color: buttonOnColor }}>Listening...</p>
          ) : (
            <p className="text-sm" style={{ color: buttonOffColor }}>Ready</p>
          )}
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
    </div>
  );
}
