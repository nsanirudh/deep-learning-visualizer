import React, { useRef, useState, useMemo } from 'react';
import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Text, Float, Environment, ContactShadows, useCursor } from '@react-three/drei';
import * as THREE from 'three';
import { ModelConfig, ProcessingStage } from '../types';

const StageMaterial = ({
  color,
  hovered,
  active,
  isData,
  isDark,
}: {
  color: string;
  hovered: boolean;
  active: boolean;
  isData: boolean;
  isDark: boolean;
}) => {
  const baseColor = useMemo(() => {
    const c = new THREE.Color(color);
    if (isDark) c.offsetHSL(0, -0.28, -0.18);
    return c;
  }, [color, isDark]);

  const activeColor = isDark ? '#d4a85a' : '#60A5FA';
  const hoveredColor = useMemo(() => new THREE.Color(baseColor).offsetHSL(0, 0.1, isDark ? 0.12 : -0.08), [baseColor, isDark]);

  const resolvedColor = active ? activeColor : hovered ? hoveredColor : baseColor;

  if (isDark) {
    return (
      <meshPhysicalMaterial
        color={resolvedColor}
        roughness={0.08}
        metalness={0}
        transmission={isData ? 0.15 : 0.38}
        thickness={isData ? 0.5 : 2.5}
        clearcoat={1}
        clearcoatRoughness={0.1}
        ior={1.45}
        emissive={hovered || active ? resolvedColor : '#000000'}
        emissiveIntensity={active ? 0.25 : hovered ? 0.15 : 0}
        transparent
        opacity={isData ? 0.82 : 0.88}
      />
    );
  }

  return (
    <meshPhysicalMaterial
      color={resolvedColor}
      roughness={isData ? 0.4 : 0.2}
      metalness={isData ? 0.15 : 0.05}
      transmission={isData ? 0 : 0.1}
      thickness={isData ? 0 : 1}
      clearcoat={isData ? 0 : 0.5}
      emissive={hovered || active ? resolvedColor : '#000000'}
      emissiveIntensity={active ? 0.2 : hovered ? 0.12 : 0}
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
  isDark: boolean;
}

const LayerBlock: React.FC<LayerBlockProps> = ({
  stage,
  onClick,
  onPointerOver,
  onPointerOut,
  isActive,
  isDark,
}) => {
  const [hovered, setHover] = useState(false);
  const groupRef = useRef<THREE.Group>(null);

  useCursor(hovered);

  // Scale up on hover instead of rotating — avoids cursor/raycast mismatch
  useFrame(() => {
    if (groupRef.current) {
      const target = (hovered || isActive) ? 1.04 : 1.0;
      const current = groupRef.current.scale.x;
      const next = THREE.MathUtils.lerp(current, target, 0.12);
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
    ? (isData ? '#6b6057' : '#9c9189')
    : (isData ? '#64748B' : '#4A5568');
  const dimColor = isDark ? '#524840' : '#718096';
  const connectorColor = isDark ? '#3a3530' : '#CBD5E0';
  const edgeOpacity = isActive ? 0.8 : hovered ? 0.6 : (isDark ? 0.35 : 0.25);
  const edgeColor = isActive
    ? (isDark ? '#d4a85a' : '#60A5FA')
    : hovered
    ? (isDark ? '#9c9189' : '#94A3B8')
    : (isData ? '#64748B' : 'grey');

  return (
    <group position={new THREE.Vector3(...stage.position)}>
      {stage.position[1] > 0 && (
        <mesh position={[0, -0.5, 0]}>
          <cylinderGeometry args={[0.02, 0.02, 1, 8]} />
          <meshStandardMaterial color={connectorColor} />
        </mesh>
      )}

      {/* Float only bobs vertically; rotationIntensity=0 keeps mesh aligned with cursor */}
      <Float speed={isData ? 0 : 1.5} rotationIntensity={0} floatIntensity={isData ? 0 : 0.08}>
        <group ref={groupRef}>
          <mesh
            onClick={(e) => { e.stopPropagation(); onClick(); }}
            onPointerOver={handleOver}
            onPointerOut={handleOut}
          >
            <boxGeometry args={[stage.dimensions[0], stage.dimensions[1], stage.dimensions[2]]} />
            <StageMaterial color={stage.color} hovered={hovered} active={isActive} isData={isData} isDark={isDark} />

            <lineSegments>
              <edgesGeometry args={[new THREE.BoxGeometry(stage.dimensions[0], stage.dimensions[1], stage.dimensions[2])]} />
              <lineBasicMaterial color={edgeColor} transparent opacity={edgeOpacity} />
            </lineSegments>

            {isSummary && (
              <group>
                <mesh position={[0, 0.2, 0]} scale={[0.95, 1, 0.95]}>
                  <boxGeometry args={[stage.dimensions[0], 0.05, stage.dimensions[2]]} />
                  <meshBasicMaterial color={isDark ? '#3a3530' : '#94A3B8'} wireframe />
                </mesh>
                <mesh position={[0, -0.2, 0]} scale={[0.95, 1, 0.95]}>
                  <boxGeometry args={[stage.dimensions[0], 0.05, stage.dimensions[2]]} />
                  <meshBasicMaterial color={isDark ? '#3a3530' : '#94A3B8'} wireframe />
                </mesh>
              </group>
            )}
          </mesh>
        </group>
      </Float>

      <group position={[stage.dimensions[0] / 2 + 0.5, 0, 0]}>
        <Text fontSize={isData ? 0.2 : 0.25} color={labelColor} anchorX="left" anchorY="bottom">
          {stage.title}
        </Text>
        <Text position={[0, -0.2, 0]} fontSize={0.15} color={dimColor} anchorX="left" anchorY="top">
          {stage.dimLabel}
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
  const wireColor = isDark ? '#3a3530' : '#CBD5E0';
  const textColor = isDark ? '#524840' : '#94A3B8';

  return (
    <group position={[0, centerY, 0]}>
      <mesh>
        <boxGeometry args={[8, height, 8]} />
        <meshBasicMaterial color={wireColor} wireframe transparent opacity={isDark ? 0.3 : 0.2} />
      </mesh>
      <Text position={[-4.2, 0, 0]} rotation={[0, 0, Math.PI / 2]} fontSize={0.4} color={textColor} anchorX="center" anchorY="bottom">
        Transformer Layer (x1)
      </Text>
    </group>
  );
};

interface VisualizerProps {
  model: ModelConfig;
  activeStageId: string | null;
  onStageSelect: (stage: ProcessingStage | null) => void;
  isDark: boolean;
}

export const Visualizer3D: React.FC<VisualizerProps> = ({ model, activeStageId, onStageSelect, isDark }) => {
  const centerY = useMemo(() => {
    if (!model || model.stages.length === 0) return 0;
    return model.stages[model.stages.length - 1].position[1] / 2;
  }, [model]);

  if (!model) return null;

  const bgColor = isDark ? '#151210' : '#F8FAFC';
  const fogColor = isDark ? '#151210' : '#F8FAFC';
  const hintBg = isDark ? 'rgba(37,34,32,0.75)' : 'rgba(255,255,255,0.6)';
  const hintText = isDark ? '#6b6057' : '#94a3b8';
  const pointLightColor = isDark ? '#d4a85a' : '#A7C7E7';

  return (
    <div className="w-full h-full relative" style={{ background: bgColor }}>
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 backdrop-blur px-4 py-1 rounded-full text-xs pointer-events-none"
        style={{ background: hintBg, color: hintText }}>
        Drag to rotate • Scroll to zoom • Click blocks for details
      </div>

      <Canvas shadows camera={{ position: [12, centerY, 16], fov: 40 }} style={{ background: bgColor }}>
        <fog attach="fog" args={[fogColor, 20, 60]} />
        <ambientLight intensity={isDark ? 0.5 : 0.8} />
        <spotLight position={[20, 40, 20]} angle={0.3} penumbra={1} intensity={isDark ? 0.7 : 1} castShadow />
        <pointLight position={[-10, 10, -10]} intensity={isDark ? 0.8 : 0.5} color={pointLightColor} />
        {isDark && <pointLight position={[10, -5, 10]} intensity={0.3} color="#c9994e" />}

        <Environment preset={isDark ? 'night' : 'city'} />

        <group position={[0, -centerY + 5, 0]}>
          <LayerGroupVisualizer stages={model.stages} isDark={isDark} />
          {model.stages.map((stage) => (
            <LayerBlock
              key={stage.id}
              stage={stage}
              isActive={activeStageId === stage.id}
              onClick={() => onStageSelect(stage)}
              onPointerOver={() => onStageSelect(stage)}
              onPointerOut={() => onStageSelect(null)}
              isDark={isDark}
            />
          ))}
        </group>

        <ContactShadows opacity={isDark ? 0.25 : 0.4} scale={40} blur={2} far={4.5} color={isDark ? '#000000' : '#475569'} />
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={6}
          maxDistance={45}
          minPolarAngle={0}
          maxPolarAngle={Math.PI}
          target={[0, centerY, 0]}
          zoomSpeed={0.7}
        />
      </Canvas>
    </div>
  );
};
