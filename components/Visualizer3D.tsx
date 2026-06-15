import React, { useRef, useState, useMemo, Suspense } from 'react';
import { Canvas, useFrame, ThreeEvent, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Float, ContactShadows, useCursor } from '@react-three/drei';
import * as THREE from 'three';
import { ModelConfig, ProcessingStage } from '../types';

// Animated flying pulse particles along the layer connection cables
const ConnectorParticles = ({ color, speed = 1.4 }: { color: string; speed?: number }) => {
  const p1Ref = useRef<THREE.Mesh>(null);
  const p2Ref = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    const time = state.clock.getElapsedTime() * speed;
    
    // Animate Y position from -1.0 to 0.0 relative to block center (the cylinder span)
    if (p1Ref.current) {
      p1Ref.current.position.y = (time % 1.0) - 1.0;
    }
    if (p2Ref.current) {
      p2Ref.current.position.y = ((time + 0.5) % 1.0) - 1.0;
    }
  });

  return (
    <group>
      <mesh ref={p1Ref}>
        <sphereGeometry args={[0.045, 12, 12]} />
        <meshBasicMaterial color={color} transparent opacity={0.8} />
      </mesh>
      <mesh ref={p2Ref}>
        <sphereGeometry args={[0.045, 12, 12]} />
        <meshBasicMaterial color={color} transparent opacity={0.8} />
      </mesh>
    </group>
  );
};

// Helper for pastel material with soft shading and custom display setups
const StageMaterial = ({
  color,
  hovered,
  active,
  isData,
  isDark,
  shadingStyle,
}: {
  color: string;
  hovered: boolean;
  active: boolean;
  isData: boolean;
  isDark: boolean;
  shadingStyle: 'cyber' | 'clay' | 'wireframe';
}) => {
  const baseColor = useMemo(() => {
    const c = new THREE.Color(color);
    if (isDark && shadingStyle !== 'wireframe') {
      // Offset slightly to render depth nicely in dark environments
      c.offsetHSL(0, -0.15, -0.1);
    }
    return c;
  }, [color, isDark, shadingStyle]);

  const activeColor = isDark ? '#818cf8' : '#3B82F6';
  const hoveredColor = useMemo(() => new THREE.Color(baseColor).offsetHSL(0, 0.05, isDark ? 0.08 : -0.05), [baseColor, isDark]);

  const resolvedColor = active ? activeColor : hovered ? hoveredColor : baseColor;

  if (shadingStyle === 'wireframe') {
    return (
      <meshBasicMaterial
        color={resolvedColor}
        wireframe
        transparent
        opacity={active ? 0.95 : hovered ? 0.75 : 0.3}
      />
    );
  }

  if (shadingStyle === 'clay') {
    return (
      <meshStandardMaterial
        color={resolvedColor}
        roughness={0.88}
        metalness={0.05}
        emissive={hovered || active ? resolvedColor : '#000000'}
        emissiveIntensity={active ? 0.35 : hovered ? 0.18 : 0}
      />
    );
  }

  // default: cyber/glass
  if (isDark) {
    return (
      <meshPhysicalMaterial
        color={resolvedColor}
        roughness={0.12}
        metalness={0.15}
        transmission={isData ? 0.15 : 0.45}
        thickness={isData ? 0.4 : 2.2}
        clearcoat={1.0}
        clearcoatRoughness={0.1}
        ior={1.48}
        emissive={hovered || active ? resolvedColor : '#000000'}
        emissiveIntensity={active ? 0.38 : hovered ? 0.2 : 0}
        transparent
        opacity={isData ? 0.78 : 0.86}
      />
    );
  }

  return (
    <meshPhysicalMaterial
      color={resolvedColor}
      roughness={isData ? 0.38 : 0.18}
      metalness={isData ? 0.15 : 0.08}
      transmission={isData ? 0.02 : 0.22}
      thickness={isData ? 0 : 1.2}
      clearcoat={isData ? 0 : 0.55}
      emissive={hovered || active ? resolvedColor : '#000000'}
      emissiveIntensity={active ? 0.28 : hovered ? 0.15 : 0}
      transparent
      opacity={isData ? 0.88 : 0.94}
    />
  );
};

interface LayerBlockProps {
  stage: ProcessingStage;
  onClick: () => void;
  onPointerOver: (e: ThreeEvent<PointerEvent>) => void;
  onPointerOut: (e: ThreeEvent<PointerEvent>) => void;
  isActive: boolean;
  isDark: boolean;
  shadingStyle: 'cyber' | 'clay' | 'wireframe';
  batchSize: number;
  seqLength: number;
  showParticles?: boolean;
}

const LayerBlock: React.FC<LayerBlockProps> = ({
  stage,
  onClick,
  onPointerOver,
  onPointerOut,
  isActive,
  isDark,
  shadingStyle,
  batchSize,
  seqLength,
  showParticles = true,
}) => {
  const [hovered, setHover] = useState(false);
  const groupRef = useRef<THREE.Group>(null);

  useCursor(hovered);

  useFrame(() => {
    if (groupRef.current) {
      const target = (hovered || isActive) ? 1.05 : 1.0;
      const current = groupRef.current.scale.x;
      const next = THREE.MathUtils.lerp(current, target, 0.14);
      groupRef.current.scale.setScalar(next);
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

  const labelColor = isDark
    ? (isData ? '#6a7ba2' : '#94a3b8')
    : (isData ? '#64748B' : '#4A5568');
  const dimColor = isDark ? '#3b4270' : '#718096';
  const connectorColor = isDark ? '#1f2438' : '#CBD5E0';
  const edgeOpacity = isActive ? 0.85 : hovered ? 0.65 : (isDark ? 0.35 : 0.25);
  const edgeColor = isActive
    ? (isDark ? '#818cf8' : '#3B82F6')
    : hovered
    ? (isDark ? '#94a3b8' : '#94A3B8')
    : (isData ? '#64748B' : '#718096');

  // Dynamic dimension text evaluation
  const formattedDimLabel = useMemo(() => {
    return stage.dimLabel
      .replace(/\bB\b/g, batchSize.toString())
      .replace(/\bS\b/g, seqLength.toString());
  }, [stage.dimLabel, batchSize, seqLength]);

  return (
    <group position={new THREE.Vector3(...stage.position)}>
      {stage.position[1] > 0 && (
        <group position={[0, -0.5, 0]}>
          <mesh>
            <cylinderGeometry args={[0.015, 0.015, 1, 8]} />
            <meshStandardMaterial color={connectorColor} roughness={0.5} />
          </mesh>
          {showParticles && shadingStyle !== 'wireframe' && (
            <ConnectorParticles color={stage.color} speed={1.3} />
          )}
        </group>
      )}

      <Float speed={isData ? 0 : 1.4} rotationIntensity={0} floatIntensity={isData ? 0 : 0.07}>
        <group ref={groupRef}>
          <mesh
            onClick={(e) => { e.stopPropagation(); onClick(); }}
            onPointerOver={handleOver}
            onPointerOut={handleOut}
          >
            <boxGeometry args={[stage.dimensions[0], stage.dimensions[1], stage.dimensions[2]]} />
            <StageMaterial
              color={stage.color}
              hovered={hovered}
              active={isActive}
              isData={isData}
              isDark={isDark}
              shadingStyle={shadingStyle}
            />

            <lineSegments>
              <edgesGeometry args={[new THREE.BoxGeometry(stage.dimensions[0], stage.dimensions[1], stage.dimensions[2])]} />
              <lineBasicMaterial color={edgeColor} transparent opacity={edgeOpacity} />
            </lineSegments>

            {isSummary && shadingStyle !== 'wireframe' && (
              <group>
                <mesh position={[0, 0.2, 0]} scale={[0.96, 1, 0.96]}>
                  <boxGeometry args={[stage.dimensions[0], 0.05, stage.dimensions[2]]} />
                  <meshBasicMaterial color={isDark ? '#1f2438' : '#94A3B8'} wireframe />
                </mesh>
                <mesh position={[0, -0.2, 0]} scale={[0.96, 1, 0.96]}>
                  <boxGeometry args={[stage.dimensions[0], 0.05, stage.dimensions[2]]} />
                  <meshBasicMaterial color={isDark ? '#1f2438' : '#94A3B8'} wireframe />
                </mesh>
              </group>
            )}
          </mesh>
        </group>
      </Float>

      <group position={[stage.dimensions[0] / 2 + 0.4, 0, 0]}>
        <Text fontSize={isData ? 0.18 : 0.24} color={labelColor} anchorX="left" anchorY="bottom">
          {stage.title}
        </Text>
        <Text position={[0, -0.16, 0]} fontSize={0.14} color={dimColor} anchorX="left" anchorY="top">
          {formattedDimLabel}
        </Text>
      </group>
    </group>
  );
};

const LayerGroupVisualizer: React.FC<{ stages: ProcessingStage[]; isDark: boolean }> = ({ stages, isDark }) => {
  const layerStages = stages.filter(s => s.group === 'layer_1');
  if (layerStages.length === 0) return null;

  let minY = Infinity;
  let maxY = -Infinity;
  layerStages.forEach(s => {
    minY = Math.min(minY, s.position[1] - s.dimensions[1] / 2);
    maxY = Math.max(maxY, s.position[1] + s.dimensions[1] / 2);
  });
  minY -= 0.5;
  maxY += 0.5;

  const height = maxY - minY;
  const centerY = minY + height / 2;
  const wireColor = isDark ? '#1f2438' : '#CBD5E0';
  const textColor = isDark ? '#6a7ba2' : '#94A3B8';

  return (
    <group position={[0, centerY, 0]}>
      <mesh>
        <boxGeometry args={[8.5, height, 8.5]} />
        <meshBasicMaterial color={wireColor} wireframe transparent opacity={isDark ? 0.22 : 0.15} />
      </mesh>
      <Text position={[-4.5, 0, 0]} rotation={[0, 0, Math.PI / 2]} fontSize={0.38} color={textColor} anchorX="center" anchorY="bottom">
        Transformer Layer (Active Block)
      </Text>
    </group>
  );
};

// Director Component to handle cinematic panning and automatic orbit rotations
const CameraDirector = ({
  autoRotate,
  autoRotateSpeed,
  centerY,
  controlsRef,
}: {
  autoRotate: boolean;
  autoRotateSpeed: number;
  centerY: number;
  controlsRef: React.MutableRefObject<any>;
}) => {
  const { camera } = useThree();
  const angleRef = useRef(0.6);

  useFrame((state, delta) => {
    if (controlsRef.current) {
      // Normal mode or passive Auto-Rotate
      if (autoRotate) {
        angleRef.current += autoRotateSpeed * 0.05 * delta;
        const radius = 17.5;
        const targetWorldY = centerY;

        controlsRef.current.target.x = THREE.MathUtils.lerp(controlsRef.current.target.x, 0, 0.08);
        controlsRef.current.target.y = THREE.MathUtils.lerp(controlsRef.current.target.y, targetWorldY, 0.08);
        controlsRef.current.target.z = THREE.MathUtils.lerp(controlsRef.current.target.z, 0, 0.08);

        const desiredCamX = Math.sin(angleRef.current) * radius;
        const desiredCamY = targetWorldY + 2;
        const desiredCamZ = Math.cos(angleRef.current) * radius;

        camera.position.x = THREE.MathUtils.lerp(camera.position.x, desiredCamX, 0.04);
        camera.position.y = THREE.MathUtils.lerp(camera.position.y, desiredCamY, 0.04);
        camera.position.z = THREE.MathUtils.lerp(camera.position.z, desiredCamZ, 0.04);

        controlsRef.current.update();
      }
    }
  });

  return null;
};

interface VisualizerProps {
  model: ModelConfig;
  activeStageId: string | null;
  onStageSelect: (stage: ProcessingStage) => void;
  onHoverStage?: (stage: ProcessingStage) => void;
  isDark: boolean;
  shadingStyle?: 'cyber' | 'clay' | 'wireframe';
  autoRotate?: boolean;
  autoRotateSpeed?: number;
  batchSize?: number;
  seqLength?: number;
  showParticles?: boolean;
}

export const Visualizer3D: React.FC<VisualizerProps> = ({
  model,
  activeStageId,
  onStageSelect,
  onHoverStage,
  isDark,
  shadingStyle = 'cyber',
  autoRotate = false,
  autoRotateSpeed = 2.0,
  batchSize = 1,
  seqLength = 2048,
  showParticles = true,
}) => {
  const controlsRef = useRef<any>(null);

  const centerY = useMemo(() => {
    if (!model || model.stages.length === 0) return 0;
    return model.stages[model.stages.length - 1].position[1] / 2;
  }, [model]);

  const activeStage = useMemo(() => {
    return model.stages.find(s => s.id === activeStageId) || null;
  }, [model, activeStageId]);

  if (!model) return null;

  const bgColor = isDark ? '#0c0e16' : '#F8FAFC';
  const fogColor = isDark ? '#0c0e16' : '#F8FAFC';
  const hintBg = isDark ? 'rgba(17,20,31,0.88)' : 'rgba(255,255,255,0.7)';
  const hintText = isDark ? '#94a3b8' : '#64748B';
  const pointLightColor = isDark ? '#818cf8' : '#A7C7E7';

  return (
    <div className="w-full h-full relative" style={{ background: bgColor }}>
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 backdrop-blur px-4 py-1.5 rounded-full text-[10px] uppercase tracking-wider font-semibold pointer-events-none shadow-sm flex items-center gap-2 border"
        style={{ background: hintBg, color: hintText, borderColor: isDark ? '#2a3252' : '#E2E8F0' }}>
        <span>Drag to rotate</span>
        <span className="text-slate-300">•</span>
        <span>Scroll to zoom</span>
        <span className="text-slate-300">•</span>
        <span>Click blocks for formulas</span>
      </div>

      <Canvas shadows camera={{ position: [11, centerY, 15], fov: 40 }} style={{ background: bgColor }}>
        <Suspense fallback={null}>
          <fog attach="fog" args={[fogColor, 18, 55]} />
          <ambientLight intensity={isDark ? 0.55 : 0.85} />
          <spotLight position={[20, 38, 20]} angle={0.28} penumbra={1} intensity={isDark ? 0.8 : 1.1} castShadow />
          <pointLight position={[-12, 12, -12]} intensity={isDark ? 0.95 : 0.6} color={pointLightColor} />
          {isDark && <pointLight position={[12, -6, 12]} intensity={0.4} color="#9ca3f4" />}

          <group position={[0, -centerY + 5, 0]}>
            <LayerGroupVisualizer stages={model.stages} isDark={isDark} />
            {model.stages.map((stage) => (
              <LayerBlock
                key={stage.id}
                stage={stage}
                isActive={activeStageId === stage.id}
                onClick={() => onStageSelect(stage)}
                onPointerOver={() => onHoverStage?.(stage)}
                onPointerOut={() => {}}
                isDark={isDark}
                shadingStyle={shadingStyle}
                batchSize={batchSize}
                seqLength={seqLength}
                showParticles={showParticles}
              />
            ))}
          </group>

          <ContactShadows opacity={isDark ? 0.35 : 0.45} scale={45} blur={2.2} far={4} color={isDark ? '#000000' : '#334155'} />

          <CameraDirector
            autoRotate={autoRotate}
            autoRotateSpeed={autoRotateSpeed}
            centerY={centerY}
            controlsRef={controlsRef}
          />
        </Suspense>

        <OrbitControls
          ref={controlsRef}
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={5}
          maxDistance={40}
          minPolarAngle={0}
          maxPolarAngle={Math.PI / 1.95}
          target={[0, centerY, 0]}
          zoomSpeed={0.85}
        />
      </Canvas>
    </div>
  );
};
