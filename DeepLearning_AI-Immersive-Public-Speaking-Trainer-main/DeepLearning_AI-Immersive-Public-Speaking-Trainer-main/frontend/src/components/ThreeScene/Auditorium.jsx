import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { 
  PerspectiveCamera, 
  MeshReflectorMaterial, 
  SpotLight,
  Float
} from '@react-three/drei';
import * as THREE from 'three';

export default function Auditorium({ isPracticeActive, volume = 0 }) {
  const spotlightRef = useRef();
  
  // Optimize: Animate spotlight based on volume
  useFrame((state) => {
    if (isPracticeActive && spotlightRef.current) {
      spotlightRef.current.intensity = 1.0 + (volume * 4);
      const t = state.clock.getElapsedTime();
      spotlightRef.current.target.position.x = Math.sin(t * 0.3) * 2;
    }
  });

  return (
    <>
      <color attach="background" args={["#050505"]} />
      
      {/* Basic Lighting */}
      <ambientLight intensity={0.15} />
      <pointLight position={[0, 10, -5]} intensity={0.5} color="#4f46e5" />
      
      <SpotLight
        ref={spotlightRef}
        position={[0, 12, -8]}
        angle={0.4}
        penumbra={0.6}
        intensity={1.2}
        castShadow
        color="#ffffff"
        attenuation={5}
        distance={30}
      />

      {/* Speaker Stage (Optimized) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, -5]} receiveShadow>
        <planeGeometry args={[20, 15]} />
        <meshStandardMaterial color="#111" roughness={0.8} metalness={0.2} />
      </mesh>

      {/* Auditorium Floor (Reflective but simple) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 15]} receiveShadow>
        <planeGeometry args={[60, 40]} />
        <MeshReflectorMaterial
          blur={[100, 100]}
          resolution={512} // Reduced resolution for performance
          mixBlur={0.5}
          mixStrength={10}
          roughness={1}
          depthScale={1}
          minDepthThreshold={0.4}
          maxDepthThreshold={1.4}
          color="#111"
          metalness={0.5}
        />
      </mesh>

      {/* Modern Stage Backdrop */}
      <mesh position={[0, 4, -12]}>
        <planeGeometry args={[25, 10]} />
        <meshStandardMaterial color="#050505" emissive="#4f46e5" emissiveIntensity={0.1} />
      </mesh>

      {/* Optimized Audience Implementation */}
      <InstancedAudience rows={6} cols={15} isActive={isPracticeActive} />

      {/* Speaker Perspective Camera */}
      <PerspectiveCamera 
        makeDefault 
        position={[0, 1.8, -7]} 
        fov={65} 
        rotation={[0, Math.PI, 0]} // Face the audience
      />
      
      {/* Subtle Camera Sway (Natural breathing effect) */}
      <CameraSway active={isPracticeActive} />
    </>
  );
}

function InstancedAudience({ rows, cols, isActive }) {
  const meshRef = useRef();
  const count = rows * cols;
  
  const dummy = useMemo(() => new THREE.Object3D(), []);
  
  useFrame((state) => {
    if (!meshRef.current) return;
    const t = state.clock.getElapsedTime();
    
    let i = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = (c - cols / 2) * 2.8;
        const z = r * 2.5 + 5;
        
        // Subtle breathing/jitter animation
        const offset = (isActive ? Math.sin(t + i) * 0.03 : 0);
        dummy.position.set(x, -0.5 + offset, z);
        dummy.scale.setScalar(0.9 + Math.sin(t * 0.2 + i) * 0.02);
        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i++, dummy.matrix);
      }
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]}>
      <capsuleGeometry args={[0.35, 0.8, 3, 6]} />
      <meshStandardMaterial color="#000" roughness={1} />
    </instancedMesh>
  );
}

function CameraSway({ active }) {
  useFrame((state) => {
    if (active) {
      const t = state.clock.getElapsedTime();
      state.camera.position.x = Math.sin(t * 0.5) * 0.05;
      state.camera.position.y = 1.8 + Math.cos(t * 0.3) * 0.02;
      // Slight rotation sway
      state.camera.rotation.z = Math.sin(t * 0.2) * 0.005;
    }
  });
  return null;
}
