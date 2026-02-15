import React, { useRef, useState, useMemo } from 'react';
import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Text, Float, Environment, ContactShadows, useCursor } from '@react-three/drei';
import * as THREE from 'three';
import { ModelConfig, ProcessingStage } from '../types';

// Helper for pastel material with soft shading
const StageMaterial = ({ color, hovered, active, isData }: { color: string; hovered: boolean; active: boolean; isData: boolean }) => {
  return (
    <meshPhysicalMaterial
      color={active ? '#60A5FA' : (hovered ? new THREE.Color(color).offsetHSL(0, 0, -0.05) : color)}
      roughness={isData ? 0.4 : 0.2}
      metalness={isData ? 0.3 : 0.1}
      transmission={isData ? 0 : 0.1} 
      thickness={isData ? 0 : 1}
      clearcoat={isData ? 0 : 0.5}
      transparent
      opacity={isData ? 0.9 : 0.95}
    />
  );
};

interface LayerBlockProps {
  stage: ProcessingStage;
  onClick: () => void;
  onPointerOver: (e: ThreeEvent<PointerEvent>) => void;
  onPointerOut: (e: ThreeEvent<PointerEvent>) => void;
  isActive: boolean;
}

const LayerBlock: React.FC<LayerBlockProps> = ({ 
  stage, 
  onClick, 
  onPointerOver, 
  onPointerOut,
  isActive 
}) => {
  const [hovered, setHover] = useState(false);
  const meshRef = useRef<THREE.Mesh>(null);
  
  useCursor(hovered);

  useFrame((state) => {
    // Subtle rotation for operations, static for data
    if (stage.category !== 'data' && meshRef.current) {
        if (hovered) {
             meshRef.current.rotation.y = THREE.MathUtils.lerp(meshRef.current.rotation.y, 0.1, 0.1);
        } else {
             meshRef.current.rotation.y = THREE.MathUtils.lerp(meshRef.current.rotation.y, 0, 0.1);
        }
    }
  });

  const handleOver = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    setHover(true);
    onPointerOver(e);
  };

  const handleOut = (e: ThreeEvent<PointerEvent>) => {
    setHover(false);
    onPointerOut(e);
  };

  const isData = stage.category === 'data';
  const isSummary = stage.category === 'summary';

  return (
    <group position={new THREE.Vector3(...stage.position)}>
      {/* Connector Line to prev layer (visual only) */}
      {stage.position[1] > 0 && (
         <mesh position={[0, -0.5, 0]}>
           <cylinderGeometry args={[0.02, 0.02, 1, 8]} />
           <meshStandardMaterial color="#CBD5E0" />
         </mesh>
      )}

      <Float speed={isData ? 0 : 2} rotationIntensity={isData ? 0 : 0.05} floatIntensity={0.1}>
        <mesh
          ref={meshRef}
          onClick={(e) => { e.stopPropagation(); onClick(); }}
          onPointerOver={handleOver}
          onPointerOut={handleOut}
        >
          {/* Summary blocks are stacked boxes visually */}
          {isSummary ? (
             <boxGeometry args={[stage.dimensions[0], stage.dimensions[1], stage.dimensions[2]]} />
          ) : (
             <boxGeometry args={[stage.dimensions[0], stage.dimensions[1], stage.dimensions[2]]} />
          )}
          
          <StageMaterial color={stage.color} hovered={hovered} active={isActive} isData={isData} />
          
          <lineSegments>
            <edgesGeometry args={[new THREE.BoxGeometry(stage.dimensions[0], stage.dimensions[1], stage.dimensions[2])]} />
            <lineBasicMaterial color={isActive ? "white" : (isData ? "#64748B" : "grey")} transparent opacity={0.3} />
          </lineSegments>
          
          {/* Extra visual for summary block to look like a stack */}
          {isSummary && (
             <group>
                <mesh position={[0, 0.2, 0]} scale={[0.95, 1, 0.95]}>
                   <boxGeometry args={[stage.dimensions[0], 0.05, stage.dimensions[2]]} />
                   <meshBasicMaterial color="#94A3B8" wireframe />
                </mesh>
                <mesh position={[0, -0.2, 0]} scale={[0.95, 1, 0.95]}>
                   <boxGeometry args={[stage.dimensions[0], 0.05, stage.dimensions[2]]} />
                   <meshBasicMaterial color="#94A3B8" wireframe />
                </mesh>
             </group>
          )}
        </mesh>
      </Float>

      {/* Label */}
      <group position={[stage.dimensions[0]/2 + 0.5, 0, 0]}>
        <Text
          fontSize={isData ? 0.2 : 0.25}
          color={isData ? "#64748B" : "#4A5568"}
          anchorX="left"
          anchorY="bottom"
        >
          {stage.title}
        </Text>
        <Text
          position={[0, -0.2, 0]}
          fontSize={0.15}
          color="#718096"
          anchorX="left"
          anchorY="top"
        >
          {stage.dimLabel}
        </Text>
      </group>
    </group>
  );
};

// Component to draw a bounding box around the single layer group
const LayerGroupVisualizer: React.FC<{ stages: ProcessingStage[] }> = ({ stages }) => {
    const layerStages = stages.filter(s => s.group === 'layer_1');
    if (layerStages.length === 0) return null;

    // Calculate bounds
    let minY = Infinity;
    let maxY = -Infinity;
    
    layerStages.forEach(s => {
        const y = s.position[1];
        const h = s.dimensions[1];
        minY = Math.min(minY, y - h/2);
        maxY = Math.max(maxY, y + h/2);
    });
    
    // Add padding
    minY -= 0.5;
    maxY += 0.5;
    
    const height = maxY - minY;
    const centerY = minY + height / 2;

    return (
        <group position={[0, centerY, 0]}>
            <mesh>
                <boxGeometry args={[8, height, 8]} />
                <meshBasicMaterial color="#CBD5E0" wireframe transparent opacity={0.2} />
            </mesh>
            <Text 
                position={[-4.2, 0, 0]} 
                rotation={[0, 0, Math.PI / 2]}
                fontSize={0.4}
                color="#94A3B8"
                anchorX="center"
                anchorY="bottom"
            >
                Transformer Layer (x1)
            </Text>
        </group>
    )
}

interface VisualizerProps {
  model: ModelConfig;
  activeStageId: string | null;
  onStageSelect: (stage: ProcessingStage | null) => void;
}

export const Visualizer3D: React.FC<VisualizerProps> = ({ model, activeStageId, onStageSelect }) => {
  // Center camera based on model height
  const centerY = useMemo(() => {
    if (!model || model.stages.length === 0) return 0;
    const last = model.stages[model.stages.length - 1];
    return last.position[1] / 2;
  }, [model]);

  if (!model) return null;

  return (
    <div className="w-full h-full bg-slate-50 cursor-move relative">
       <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 bg-white/50 backdrop-blur px-4 py-1 rounded-full text-xs text-slate-500 pointer-events-none">
          Drag to rotate • Scroll to zoom • Click blocks for details
       </div>

      <Canvas shadows camera={{ position: [12, centerY, 16], fov: 40 }}>
        <fog attach="fog" args={['#F8FAFC', 20, 60]} />
        <ambientLight intensity={0.8} />
        <spotLight position={[20, 40, 20]} angle={0.3} penumbra={1} intensity={1} castShadow />
        <pointLight position={[-10, 10, -10]} intensity={0.5} color="#A7C7E7" />
        
        <Environment preset="city" />

        <group position={[0, -centerY + 5, 0]}>
          <LayerGroupVisualizer stages={model.stages} />
          
          {model.stages.map((stage) => (
            <LayerBlock
              key={stage.id}
              stage={stage}
              isActive={activeStageId === stage.id}
              onClick={() => onStageSelect(stage)}
              onPointerOver={() => onStageSelect(stage)}
              onPointerOut={() => onStageSelect(null)}
            />
          ))}
        </group>

        <ContactShadows opacity={0.4} scale={40} blur={2} far={4.5} />
        <OrbitControls 
          enablePan={true} 
          enableZoom={true}
          enableRotate={true}
          minPolarAngle={0} 
          maxPolarAngle={Math.PI}
          target={[0, centerY, 0]}
        />
      </Canvas>
    </div>
  );
};