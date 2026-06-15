import React, { useState, useEffect, useMemo } from 'react';
import { ModelType, ProcessingStage } from './types';
import { MODELS } from './constants';
import { Visualizer3D } from './components/Visualizer3D';
import { MathPanel } from './components/MathPanel';
import { NotesPanel } from './components/NotesPanel';
import { motion, AnimatePresence } from 'framer-motion';

const SunIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <circle cx="12" cy="12" r="5" />
    <line x1="12" y1="1" x2="12" y2="3" />
    <line x1="12" y1="21" x2="12" y2="23" />
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
    <line x1="1" y1="12" x2="3" y2="12" />
    <line x1="21" y1="12" x2="23" y2="12" />
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
  </svg>
);

const MoonIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
  </svg>
);

const SidebarItem = ({
  label,
  value,
  sub,
  isDark,
}: {
  label: string;
  value: string | number;
  sub?: string;
  isDark: boolean;
}) => (
  <div className={`flex flex-col py-2 border-b last:border-0 ${isDark ? 'border-[#2a3252]' : 'border-slate-100'}`}>
    <span className={`text-[10px] uppercase font-bold tracking-wider mb-0.5 ${isDark ? 'text-[#6a7ba2]' : 'text-slate-400'}`}>{label}</span>
    <span className={`text-xs font-semibold font-mono ${isDark ? 'text-[#e2e8f0]' : 'text-slate-700'}`}>{value}</span>
    {sub && <span className={`text-[9px] font-medium leading-relaxed mt-0.5 ${isDark ? 'text-[#6a7ba2]' : 'text-slate-400'}`}>{sub}</span>}
  </div>
);

const InfoItem = ({
  label,
  value,
  detail,
  isDark,
}: {
  label: string;
  value: string;
  detail?: string;
  isDark: boolean;
}) => (
  <div className={`flex flex-col py-2 border-b last:border-0 ${isDark ? 'border-[#2a3252]' : 'border-slate-100'}`}>
    <span className={`text-[10px] uppercase font-bold tracking-wider mb-0.5 ${isDark ? 'text-[#6a7ba2]' : 'text-slate-400'}`}>{label}</span>
    <span className={`text-xs font-semibold font-mono ${isDark ? 'text-[#e2e8f0]' : 'text-slate-700'}`}>{value}</span>
    {detail && <span className={`text-[9px] leading-relaxed mt-0.5 ${isDark ? 'text-[#6a7ba2]' : 'text-slate-400'}`}>{detail}</span>}
  </div>
);

const SidebarSection = ({
  title,
  isDark,
  children,
}: {
  title: string;
  isDark: boolean;
  children: React.ReactNode;
}) => (
  <div
    className={`rounded-xl p-4 mb-4 ${isDark ? '' : 'bg-slate-50 border border-slate-100'}`}
    style={isDark ? { background: '#1a1d2e', border: '1px solid #2a3252' } : {}}
  >
    <h3 className={`text-[10px] font-extrabold uppercase tracking-widest mb-3 ${isDark ? 'text-[#6366f1]' : 'text-slate-800'}`}>{title}</h3>
    {children}
  </div>
);

// Dynamic Math VRAM Stats calculator
const computeDynamicVram = (
  modelType: ModelType,
  B: number,
  S: number,
  quant: 'FP32' | 'FP16' | 'INT8' | 'INT4',
  mode: 'inference' | 'training',
  opt: 'AdamW' | 'SGD'
) => {
  const bytes = quant === 'FP32' ? 4 : quant === 'FP16' ? 2 : quant === 'INT8' ? 1 : 0.5;
  
  if (modelType === ModelType.TRANSFORMER_7B || modelType === ModelType.PERSONA_PLEX) {
    const N = modelType === ModelType.TRANSFORMER_7B ? 7.0e9 : 7.2e9;
    
    // Weight VRAM
    const weightVram = (N * bytes) / 1e9; // GB
    
    // Gradients
    const gradsVram = mode === 'training' ? (N * bytes) / 1e9 : 0;
    
    // Optimizer States (FP32 moments)
    let optVram = 0;
    if (mode === 'training') {
      const optBytes = opt === 'AdamW' ? 8 : 4; 
      optVram = (N * optBytes) / 1e9;
    }
    
    // Master Weights (often FP32)
    const masterVram = (mode === 'training' && quant !== 'FP32') ? (N * 4) / 1e9 : 0;
    
    // KV Cache VRAM (2 * L * H * B * S * bytes_inference)
    const L = 32;
    const H = 4096;
    const kvCacheBytes = 2; // Keep as standard FP16 for activation stability
    const kvVram = (2 * L * H * B * S * kvCacheBytes) / 1e9; // GB
    
    // Activation memory approximation
    const actFactor = mode === 'training' ? 0.35 : 0.02;
    const actVram = (B * S * H * L * actFactor) / 1e6; // MB scale
    
    const totalGB = weightVram + gradsVram + optVram + masterVram + kvVram + (actVram / 1000);

    const breakdown = [
      { label: `Model Weights (${quant})`, value: `${weightVram.toFixed(2)} GB`, detail: `N × ${bytes}B per parameter` },
    ];
    
    if (mode === 'training') {
      breakdown.push({ label: `Gradients (${quant})`, value: `${gradsVram.toFixed(2)} GB`, detail: `Stored backprop nodes` });
      if (optVram > 0) {
        breakdown.push({ label: `Optimizer (${opt})`, value: `${optVram.toFixed(2)} GB`, detail: `${opt === 'AdamW' ? 'm & v moments (FP32)' : 'momentum state (FP32)'}` });
      }
      if (masterVram > 0) {
        breakdown.push({ label: `FP32 Master Weights`, value: `${masterVram.toFixed(2)} GB`, detail: `Precision master parameters` });
      }
    } else {
      breakdown.push({ label: `KV Cache Buffer`, value: `${(kvVram * 1024).toFixed(1)} MB`, detail: `2 × L × H × B × S × 16-bit Cache` });
    }
    
    breakdown.push({ label: `Activation Frames`, value: `${actVram.toFixed(1)} MB`, detail: `Forward tracking states` });
    
    return {
      totalParams: modelType === ModelType.TRANSFORMER_7B ? "~7.0 Billion" : "~7.2 Billion",
      paramsPerLayer: "~202 Million",
      totalvram: `${totalGB.toFixed(2)} GB`,
      breakdown
    };
  } else {
    // Wav2Vec 2.0 Base (95M params)
    const N = 95.0e6;
    const weightVram = (N * bytes) / 1e6; // MB
    const gradsVram = mode === 'training' ? (N * bytes) / 1e6 : 0;
    
    let optVram = 0;
    if (mode === 'training') {
      const optBytes = opt === 'AdamW' ? 8 : 4;
      optVram = (N * optBytes) / 1e6;
    }
    const masterVram = (mode === 'training' && quant !== 'FP32') ? (N * 4) / 1e6 : 0;
    
    const actFactor = mode === 'training' ? 0.12 : 0.01;
    const actVram = (B * S * 768 * 12 * actFactor) / 1e6; // MB
    
    const totalMB = weightVram + gradsVram + optVram + masterVram + actVram;
    const totalStr = totalMB > 1024 ? `${(totalMB / 1024).toFixed(2)} GB` : `${totalMB.toFixed(1)} MB`;

    const breakdown = [
      { label: `Model Weights (${quant})`, value: `${weightVram.toFixed(1)} MB`, detail: `N × ${bytes}B per parameter` },
    ];
    
    if (mode === 'training') {
      breakdown.push({ label: `Gradients (${quant})`, value: `${gradsVram.toFixed(1)} MB`, detail: `Temporal backward grad gradients` });
      if (optVram > 0) {
        breakdown.push({ label: `Optimizer (${opt})`, value: `${optVram.toFixed(1)} MB`, detail: `Optimizer memory moments` });
      }
      if (masterVram > 0) {
        breakdown.push({ label: `FP32 Master Weights`, value: `${masterVram.toFixed(1)} MB`, detail: `Original float copy` });
      }
    }
    breakdown.push({ label: `Activation Frames`, value: `${actVram.toFixed(1)} MB`, detail: `Acoustic filter maps` });
    
    return {
      totalParams: "~95 Million",
      paramsPerLayer: "~7.1 Million",
      totalvram: totalStr,
      breakdown
    };
  }
};

const App = () => {
  const [selectedModelType, setSelectedModelType] = useState<ModelType>(ModelType.TRANSFORMER_7B);
  const [activeStage, setActiveStage] = useState<ProcessingStage | null>(null);
  const [mathDisplayStage, setMathDisplayStage] = useState<ProcessingStage | null>(null);
  const [isDark, setIsDark] = useState(true);

  // --- Sequence & Hardware States ---
  const [runningMode, setRunningMode] = useState<'inference' | 'training'>('inference');
  const [batchSize, setBatchSize] = useState<number>(4);
  const [seqLength, setSeqLength] = useState<number>(2048);
  const [quantization, setQuantization] = useState<'FP32' | 'FP16' | 'INT8' | 'INT4'>('FP16');
  const [optimizer, setOptimizer] = useState<'AdamW' | 'SGD'>('AdamW');

  // --- Visual Controls ---
  const [shadingStyle, setShadingStyle] = useState<'cyber' | 'clay' | 'wireframe'>('cyber');
  const [showParticles, setShowParticles] = useState<boolean>(true);
  const [autoRotate, setAutoRotate] = useState<boolean>(false);
  const [showNotes, setShowNotes] = useState<boolean>(false);

  // --- Playback States ---
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [activePlaybackIndex, setActivePlaybackIndex] = useState<number>(-1);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1200); // ms per step

  const currentModel = MODELS[selectedModelType];

  // Sequencer loop
  useEffect(() => {
    let interval: any = null;
    if (isPlaying) {
      if (activePlaybackIndex === -1) {
        setActivePlaybackIndex(0);
        setActiveStage(currentModel.stages[0]);
        setMathDisplayStage(currentModel.stages[0]);
      }
      interval = setInterval(() => {
        setActivePlaybackIndex((prev) => {
          const next = prev + 1 >= currentModel.stages.length ? 0 : prev + 1;
          const nextStage = currentModel.stages[next];
          setActiveStage(nextStage);
          setMathDisplayStage(nextStage);
          return next;
        });
      }, playbackSpeed);
    } else {
      if (interval) clearInterval(interval);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying, playbackSpeed, currentModel.stages, activePlaybackIndex]);

  // Handle stage change manually or on model shift
  useEffect(() => {
    setIsPlaying(false);
    setActivePlaybackIndex(-1);
    setActiveStage(null);
    setMathDisplayStage(null);
  }, [selectedModelType]);

  const handleStageClick = (stage: ProcessingStage) => {
    setActiveStage(stage);
    setMathDisplayStage(stage);
    const idx = currentModel.stages.findIndex(s => s.id === stage.id);
    if (idx !== -1) setActivePlaybackIndex(idx);
  };

  const handleStageHover = (stage: ProcessingStage) => {
    setMathDisplayStage(stage);
  };

  const dynamicStats = useMemo(() => {
    return computeDynamicVram(selectedModelType, batchSize, seqLength, quantization, runningMode, optimizer);
  }, [selectedModelType, batchSize, seqLength, quantization, runningMode, optimizer]);

  return (
    <div className={`flex h-screen w-screen overflow-hidden ${isDark ? 'text-[#e2e8f0]' : 'bg-slate-50 text-slate-800'}`} style={isDark ? { background: '#0c0e16' } : {}}>
      
      {/* Dynamic Sidebar */}
      <div className="w-80 md:w-96 flex-shrink-0 flex flex-col z-20 shadow-xl border-r" 
        style={isDark ? { background: '#11141f', borderColor: '#2a3252' } : { background: 'white', borderColor: '#e2e8f0' }}>
        
        {/* Header Block */}
        <div className="p-5 flex items-start justify-between" style={isDark ? { borderBottom: '1px solid #2a3252' } : { borderBottom: '1px solid #f1f5f9' }}>
          <div>
            <h1 className="text-lg font-bold bg-clip-text text-transparent" 
              style={isDark ? { backgroundImage: 'linear-gradient(to right, #818cf8, #a5b4fc)' } : { backgroundImage: 'linear-gradient(to right, #334155, #475569)' }}>
              LayerGraph 3D
            </h1>
            <p className="text-[10px] uppercase font-bold tracking-widest mt-0.5" style={isDark ? { color: '#6a7ba2' } : { color: '#94a3b8' }}>Neural Hardware Estimator</p>
          </div>
          <button onClick={() => setIsDark(!isDark)} className="p-2 rounded-lg border transition-all" 
            style={isDark ? { background: '#1a1d2e', color: '#94a3b8', borderColor: '#2a3252' } : { background: '#f1f5f9', color: '#475569', borderColor: '#e2e8f0' }} 
            title={isDark ? "Light theme" : "Dark theme"}>
            {isDark ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>

        {/* Model Tabs Selection */}
        <div className="p-4 bg-opacity-40" style={isDark ? { background: '#0c0e16' } : { background: '#f8fafc' }}>
          <div className="flex p-1 rounded-lg gap-1 border" style={isDark ? { background: '#1a1d2e', borderColor: '#2a3252' } : { background: '#f1f5f9', borderColor: '#e2e8f0' }}>
            {([
              [ModelType.TRANSFORMER_7B, 'Llama-7B'],
              [ModelType.PERSONA_PLEX, 'PersonaPlex'],
              [ModelType.WAV2VEC_100M, 'Wav2Vec-2'],
            ] as [ModelType, string][]).map(([type, label]) => (
              <button key={type} onClick={() => setSelectedModelType(type)} className="flex-1 py-1.5 text-[10px] md:text-xs font-semibold rounded transition-all"
                style={isDark ? {
                  background: selectedModelType === type ? '#1f2438' : 'transparent',
                  color: selectedModelType === type ? '#e2e8f0' : '#6a7ba2',
                  boxShadow: selectedModelType === type ? '0 1px 3px rgba(0,0,0,0.3)' : 'none'
                } : {
                  background: selectedModelType === type ? '#ffffff' : 'transparent',
                  color: selectedModelType === type ? '#334155' : '#64748b',
                  boxShadow: selectedModelType === type ? '0 1px 2px rgba(0,0,0,0.05)' : 'none'
                }}>
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Left Side scrollable panels */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          
          {/* Active Model Title */}
          <div>
            <span className="text-[9px] font-extrabold uppercase px-1.5 py-0.5 rounded tracking-wide border" 
              style={isDark ? { color: '#6366f1', borderColor: '#1f2438', background: '#1a1d2e' } : { color: '#3b82f6', borderColor: '#e2e8f0', background: '#f8fafc' }}>
              {currentModel.id === ModelType.TRANSFORMER_7B ? 'LLM DECODER' : currentModel.id === ModelType.PERSONA_PLEX ? 'SPEECH-TO-SPEECH' : 'SPEECH LEARNER'}
            </span>
            <h2 className="text-base font-semibold mt-2" style={isDark ? { color: '#e2e8f0' } : { color: '#1e293b' }}>{currentModel.name}</h2>
            <p className="text-xs leading-relaxed mt-1" style={isDark ? { color: '#94a3b8' } : { color: '#64748b' }}>{currentModel.description}</p>
          </div>

          {/* Sequence & Hardware Optimizer sliders dashboard */}
          <SidebarSection title="Simulator Configurations" isDark={isDark}>
            {/* Running Mode Selector */}
            <div className="mb-3">
              <label className="text-[9px] uppercase tracking-wider font-extrabold block mb-1.5" style={isDark ? { color: '#6a7ba2' } : { color: '#94a3b8' }}>Engine Mode</label>
              <div className="flex rounded border p-0.5 gap-0.5" style={isDark ? { background: '#11141f', borderColor: '#2a3252' } : { background: 'white', borderColor: '#e2e8f0' }}>
                <button onClick={() => setRunningMode('inference')} className="flex-1 py-1 text-[10px] font-bold rounded"
                  style={{
                    background: runningMode === 'inference' ? (isDark ? '#1f2438' : '#475569') : 'transparent',
                    color: runningMode === 'inference' ? '#e2e8f0' : (isDark ? '#3b4270' : '#94a3b8')
                  }}>
                  Inference Mode
                </button>
                <button onClick={() => setRunningMode('training')} className="flex-1 py-1 text-[10px] font-bold rounded"
                  style={{
                    background: runningMode === 'training' ? (isDark ? '#1f2438' : '#475569') : 'transparent',
                    color: runningMode === 'training' ? '#e2e8f0' : (isDark ? '#3b4270' : '#94a3b8')
                  }}>
                  Training Mode
                </button>
              </div>
            </div>

            {/* Slider: Batch Size */}
            <div className="mb-3">
              <div className="flex justify-between items-center text-[10px] block mb-1">
                <span className="uppercase tracking-wider font-bold" style={isDark ? { color: '#6a7ba2' } : { color: '#94a3b8' }}>Batch Size (B)</span>
                <span className="font-mono font-bold" style={isDark ? { color: '#818cf8' } : { color: '#2563eb' }}>{batchSize}</span>
              </div>
              <input type="range" min="1" max="32" step={batchSize > 8 ? 4 : 1} value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value))}
                className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer" style={isDark ? { background: '#2a3252' } : {}} />
            </div>

            {/* Slider: Sequence Length */}
            <div className="mb-3">
              <div className="flex justify-between items-center text-[10px] block mb-1">
                <span className="uppercase tracking-wider font-bold" style={isDark ? { color: '#6a7ba2' } : { color: '#94a3b8' }}>Seq. Length (S)</span>
                <span className="font-mono font-bold" style={isDark ? { color: '#818cf8' } : { color: '#2563eb' }}>{seqLength} tokens</span>
              </div>
              <input type="range" min="256" max="8192" step="256" value={seqLength} onChange={(e) => setSeqLength(parseInt(e.target.value))}
                className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer" style={isDark ? { background: '#2a3252' } : {}} />
            </div>

            {/* Weight precision format picker */}
            <div className="mb-3">
              <label className="text-[9px] uppercase tracking-wider font-extrabold block mb-1" style={isDark ? { color: '#6a7ba2' } : { color: '#94a3b8' }}>Quantization Precision</label>
              <div className="grid grid-cols-4 gap-1 p-0.5 rounded border" style={isDark ? { background: '#11141f', borderColor: '#2a3252' } : { background: '#ffffff', borderColor: '#e2e8f0' }}>
                {(['FP32', 'FP16', 'INT8', 'INT4'] as const).map((q) => (
                  <button key={q} onClick={() => setQuantization(q)} className="py-1 text-[9px] font-extrabold rounded"
                    style={{
                      background: quantization === q ? (isDark ? '#1f2438' : '#475569') : 'transparent',
                      color: quantization === q ? '#e2e8f0' : (isDark ? '#6a7ba2' : '#94a3b8')
                    }}>
                    {q}
                  </button>
                ))}
              </div>
            </div>

            {/* Optimizer selector (only visible during training) */}
            {runningMode === 'training' && (
              <div className="mt-2.5">
                <label className="text-[9px] uppercase tracking-wider font-extrabold block mb-1" style={isDark ? { color: '#6a7ba2' } : { color: '#94a3b8' }}>Weights Optimizer</label>
                <div className="flex rounded border p-0.5 gap-0.5" style={isDark ? { background: '#11141f', borderColor: '#2a3252' } : { background: '#ffffff', borderColor: '#e2e8f0' }}>
                  {(['AdamW', 'SGD'] as const).map((opt) => (
                    <button key={opt} onClick={() => setOptimizer(opt)} className="flex-1 py-1 text-[9px] font-bold rounded"
                      style={{
                        background: optimizer === opt ? (isDark ? '#1f2438' : '#475569') : 'transparent',
                        color: optimizer === opt ? '#e2e8f0' : (isDark ? '#6a7ba2' : '#94a3b8')
                      }}>
                      {opt === 'AdamW' ? 'AdamW (8 bytes/param FP32)' : 'SGD Momentum (4 bytes)'}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </SidebarSection>

          {/* Training / Inference Dynamic specs */}
          <SidebarSection title="Engine Specification Summary" isDark={isDark}>
            <SidebarItem isDark={isDark} label="Model Parameters" value={dynamicStats.totalParams} />
            <SidebarItem isDark={isDark} label="Weights Per Block" value={dynamicStats.paramsPerLayer} />
            <SidebarItem isDark={isDark} label="Dynamic Total VRAM" value={dynamicStats.totalvram} sub="Based on active model configurations" />
          </SidebarSection>

          {/* Real interactive VRAM analysis */}
          <SidebarSection title="Dynamic Memory Breakdown" isDark={isDark}>
            {dynamicStats.breakdown.map((item, i) => (
              <InfoItem key={i} isDark={isDark} label={item.label} value={item.value} detail={item.detail} />
            ))}
          </SidebarSection>

          {/* Architecture insights */}
          <SidebarSection title="Architectural Blueprint Notes" isDark={isDark}>
            <ul className="space-y-2.5">
              {currentModel.architectureNotes.map((note, i) => (
                <li key={i} className="flex gap-2 text-[11px] leading-relaxed" style={isDark ? { color: '#94a3b8' } : { color: '#64748b' }}>
                  <span style={isDark ? { color: '#6366f1', flexShrink: 0 } : { color: '#3b82f6', flexShrink: 0 }}>—</span>
                  <span>{note}</span>
                </li>
              ))}
            </ul>
          </SidebarSection>

        </div>

        <div className="p-4 text-[9px] tracking-wider text-center uppercase font-semibold block" style={isDark ? { borderTop: '1px solid #2a3252', color: '#6a7ba2' } : { borderTop: '1px solid #f1f5f9', color: '#94a3b8' }}>
          Real-time WebGL Engine • Three.js LayerGraph
        </div>
      </div>

      {/* Primary area: 3D canvas + notes as proper flex columns */}
      <div className="flex-1 flex h-full overflow-hidden">
      <div className="flex-1 relative h-full min-w-0">
        
        {/* Playback Sequence Controller Overlay */}
        <div className="absolute top-6 left-6 z-10 flex flex-wrap items-center gap-2 max-w-[calc(100%-12px)]">
          
          {/* Main Action Controllers */}
          <div className="backdrop-blur-md rounded-xl p-2.5 border flex items-center gap-3.5 shadow-lg" 
            style={isDark ? { background: 'rgba(17,20,31,0.88)', borderColor: '#2a3252' } : { background: 'rgba(255,255,255,0.9)', borderColor: '#e2e8f0' }}>
            <div className="flex items-center gap-1.5">
              {/* Prev */}
              <button onClick={() => {
                setIsPlaying(false);
                const nextIdx = activePlaybackIndex - 1 < 0 ? currentModel.stages.length - 1 : activePlaybackIndex - 1;
                const nextStage = currentModel.stages[nextIdx];
                setActivePlaybackIndex(nextIdx);
                setActiveStage(nextStage);
                setMathDisplayStage(nextStage);
              }} className="p-1.5 rounded-lg border hover:-translate-y-0.5 transition-all text-sm"
                style={isDark ? { background: '#1a1d2e', borderColor: '#2a3252' } : { background: '#f8fafc', borderColor: '#e2e8f0' }} title="Previous step">
                ◀
              </button>

              {/* Play / Pause button */}
              <button onClick={() => setIsPlaying(!isPlaying)} className="px-3 py-1.5 font-extrabold rounded-lg border text-[11px] tracking-wider uppercase transition-all"
                style={isDark ? {
                  background: isPlaying ? '#818cf8' : '#1a1d2e',
                  color: isPlaying ? '#0c0e16' : '#e2e8f0',
                  borderColor: '#2a3252'
                } : {
                  background: isPlaying ? '#475569' : '#0f172a',
                  color: '#ffffff',
                  borderColor: '#e2e8f0'
                }}>
                {isPlaying ? '⏸ Pause Flow' : '▶ Play Flow'}
              </button>

              {/* Next */}
              <button onClick={() => {
                setIsPlaying(false);
                const nextIdx = activePlaybackIndex + 1 >= currentModel.stages.length ? 0 : activePlaybackIndex + 1;
                const nextStage = currentModel.stages[nextIdx];
                setActivePlaybackIndex(nextIdx);
                setActiveStage(nextStage);
                setMathDisplayStage(nextStage);
              }} className="p-1.5 rounded-lg border hover:-translate-y-0.5 transition-all text-sm"
                style={isDark ? { background: '#1a1d2e', borderColor: '#2a3252' } : { background: '#f8fafc', borderColor: '#e2e8f0' }} title="Next step">
                ▶
              </button>
            </div>

            {/* Steps indicator */}
            <div className="text-[10px] uppercase tracking-wider font-extrabold flex items-center gap-1">
              <span>Step</span>
              <span className="font-mono font-bold text-xs" style={isDark ? { color: '#818cf8' } : { color: '#1e293b' }}>
                {activePlaybackIndex === -1 ? '—' : activePlaybackIndex + 1}
              </span>
              <span className="text-slate-400">/</span>
              <span>{currentModel.stages.length}</span>
            </div>
          </div>

          {/* Quick Display Customizer Settings */}
          <div className="backdrop-blur-md rounded-xl p-2.5 border flex items-center gap-2.5 shadow-lg" 
            style={isDark ? { background: 'rgba(17,20,31,0.88)', borderColor: '#2a3252' } : { background: 'rgba(255,255,255,0.9)', borderColor: '#e2e8f0' }}>
            
            {/* Shading styles toggle */}
            <div className="flex gap-0.5 p-0.5 border rounded-lg" style={isDark ? { background: '#0c0e16', borderColor: '#2a3252' } : { background: '#f1f5f9', borderColor: '#e2e8f0' }}>
              {(['cyber', 'clay', 'wireframe'] as const).map((style) => (
                <button key={style} onClick={() => setShadingStyle(style)} className="px-2 py-1 text-[9px] font-bold rounded capitalize"
                  style={{
                    background: shadingStyle === style ? (isDark ? '#1f2438' : '#475569') : 'transparent',
                    color: shadingStyle === style ? '#e2e8f0' : (isDark ? '#6a7ba2' : '#94a3b8')
                  }}>
                  {style}
                </button>
              ))}
            </div>

            {/* Quick action triggers */}
            <div className="flex items-center gap-3">
              {/* Flow particles */}
              <label className="flex items-center gap-1.5 cursor-pointer select-none">
                <input type="checkbox" checked={showParticles} onChange={(e) => setShowParticles(e.target.checked)} className="rounded border-gray-300 w-3 h-3 text-amber-500 rounded focus:ring-0" />
                <span className="text-[10px] font-bold uppercase tracking-wider" style={isDark ? { color: '#94a3b8' } : { color: '#64748b' }}>Flow Lines</span>
              </label>

              {/* Passive rotation */}
              <label className="flex items-center gap-1.5 cursor-pointer select-none">
                <input type="checkbox" checked={autoRotate} onChange={(e) => setAutoRotate(e.target.checked)} className="rounded border-gray-300 w-3 h-3 text-amber-500 rounded focus:ring-0" />
                <span className="text-[10px] font-bold uppercase tracking-wider" style={isDark ? { color: '#94a3b8' } : { color: '#64748b' }}>Orbit</span>
              </label>
            </div>

            {/* Notes toggle */}
            <button
              onClick={() => setShowNotes(!showNotes)}
              className="px-2.5 py-1.5 text-[9px] font-extrabold rounded-lg border uppercase tracking-wider transition-all"
              style={showNotes ? {
                background: isDark ? '#6366f1' : '#4f46e5',
                color: '#ffffff',
                borderColor: isDark ? '#818cf8' : '#4f46e5',
              } : {
                background: isDark ? '#1a1d2e' : '#f1f5f9',
                color: isDark ? '#818cf8' : '#4f46e5',
                borderColor: isDark ? '#2a3252' : '#e2e8f0',
              }}>
              {showNotes ? '✕ Notes' : '≡ Notes'}
            </button>
          </div>

        </div>

        {/* 3D view render node context */}
        <Visualizer3D
          model={currentModel}
          activeStageId={activeStage?.id || null}
          onStageSelect={handleStageClick}
          onHoverStage={handleStageHover}
          isDark={isDark}
          shadingStyle={shadingStyle}
          autoRotate={autoRotate}
          batchSize={batchSize}
          seqLength={seqLength}
          showParticles={showParticles}
        />
        
        {/* Math details overlay — hide when notes panel is open */}
        {!showNotes && (
          <MathPanel stage={mathDisplayStage} isDark={isDark} batchSize={batchSize} seqLength={seqLength} />
        )}
      </div>

        {/* Architecture Notes — proper flex column, not an overlay */}
        <NotesPanel
          modelType={selectedModelType}
          isDark={isDark}
          isOpen={showNotes}
          onClose={() => setShowNotes(false)}
        />
      </div>
    </div>
  );
};

export default App;
