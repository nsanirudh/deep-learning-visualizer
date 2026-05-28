
import React, { useState } from 'react';
import { ModelType, ProcessingStage } from './types';
import { MODELS } from './constants';
import { Visualizer3D } from './components/Visualizer3D';
import { MathPanel } from './components/MathPanel';
import { motion } from 'framer-motion';

const SunIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
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
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
  </svg>
);

// Light mode uses original slate palette; dark mode uses warm charcoal
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
  <div className={`flex flex-col py-2 border-b last:border-0 ${isDark ? 'border-[#332e2a]' : 'border-slate-100'}`}>
    <span className={`text-xs uppercase tracking-wider mb-0.5 ${isDark ? 'text-[#6b6057]' : 'text-slate-400'}`}>{label}</span>
    <span className={`text-sm font-medium font-mono ${isDark ? 'text-[#f0ebe4]' : 'text-slate-700'}`}>{value}</span>
    {sub && <span className={`text-[10px] ${isDark ? 'text-[#6b6057]' : 'text-slate-400'}`}>{sub}</span>}
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
  <div className={`flex flex-col py-2 border-b last:border-0 ${isDark ? 'border-[#332e2a]' : 'border-slate-100'}`}>
    <span className={`text-xs uppercase tracking-wider mb-0.5 ${isDark ? 'text-[#6b6057]' : 'text-slate-400'}`}>{label}</span>
    <span className={`text-sm font-semibold font-mono ${isDark ? 'text-[#f0ebe4]' : 'text-slate-700'}`}>{value}</span>
    {detail && <span className={`text-[10px] leading-relaxed mt-0.5 ${isDark ? 'text-[#6b6057]' : 'text-slate-400'}`}>{detail}</span>}
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
    style={isDark ? { background: '#252220', border: '1px solid #332e2a' } : {}}
  >
    <h3 className={`text-xs font-bold uppercase mb-3 ${isDark ? 'text-[#c9994e]' : 'text-slate-800'}`}>{title}</h3>
    {children}
  </div>
);

const App = () => {
  const [selectedModelType, setSelectedModelType] = useState<ModelType>(ModelType.TRANSFORMER_7B);
  const [activeStage, setActiveStage] = useState<ProcessingStage | null>(null);
  const [isDark, setIsDark] = useState(false);

  const currentModel = MODELS[selectedModelType];

  if (isDark) {
    return (
      <div className="flex h-screen w-screen overflow-hidden" style={{ background: '#151210' }}>
        {/* Dark sidebar */}
        <div className="w-80 md:w-96 flex-shrink-0 flex flex-col z-20 shadow-lg" style={{ background: '#1c1916', borderRight: '1px solid #332e2a' }}>
          <div className="p-6 flex items-start justify-between" style={{ borderBottom: '1px solid #332e2a' }}>
            <div>
              <h1 className="text-xl font-bold bg-clip-text text-transparent" style={{ backgroundImage: 'linear-gradient(to right, #d4a85a, #f0d89a)' }}>
                LayerGraph
              </h1>
              <p className="text-xs mt-1" style={{ color: '#6b6057' }}>Deep Learning Architecture Explorer</p>
            </div>
            <button onClick={() => setIsDark(false)} className="mt-0.5 p-2 rounded-lg transition-colors" style={{ background: '#252220', color: '#9c9189', border: '1px solid #332e2a' }} title="Switch to light mode">
              <SunIcon />
            </button>
          </div>

          <div className="p-4" style={{ background: '#151210' }}>
            <div className="flex p-1 rounded-lg gap-1" style={{ background: '#252220' }}>
              {([
                [ModelType.TRANSFORMER_7B, '7B Transf.'],
                [ModelType.PERSONA_PLEX, 'PersonaPlex'],
                [ModelType.WAV2VEC_100M, 'Wav2Vec 2'],
              ] as [ModelType, string][]).map(([type, label]) => (
                <button key={type} onClick={() => setSelectedModelType(type)} className="flex-1 py-2 text-[10px] md:text-xs font-semibold rounded-md transition-all"
                  style={{ background: selectedModelType === type ? '#3a3530' : 'transparent', color: selectedModelType === type ? '#f0ebe4' : '#6b6057', boxShadow: selectedModelType === type ? '0 1px 3px rgba(0,0,0,0.3)' : 'none' }}>
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-6 scrollbar-hide">
            <motion.div key={selectedModelType} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.3 }}>
              <div className="mb-5">
                <h2 className="text-lg font-semibold mb-2" style={{ color: '#f0ebe4' }}>{currentModel.name}</h2>
                <p className="text-sm leading-relaxed" style={{ color: '#9c9189' }}>{currentModel.description}</p>
              </div>

              <SidebarSection title="Training Specs" isDark>
                <SidebarItem isDark label="Hidden Size (H)" value={currentModel.specs.hiddenSize} />
                <SidebarItem isDark label="Layers (L)" value={currentModel.specs.layers} />
                <SidebarItem isDark label="Attention Heads" value={currentModel.specs.heads} sub={`Head Dim: ${currentModel.specs.hiddenSize / currentModel.specs.heads}`} />
                <SidebarItem isDark label="FFN / MLP Size" value={currentModel.specs.ffnSize} />
                {currentModel.specs.convLayers && <SidebarItem isDark label="CNN Layers" value={currentModel.specs.convLayers} />}
              </SidebarSection>

              <SidebarSection title="Memory & Params" isDark>
                <SidebarItem isDark label="Params / Layer" value={currentModel.specs.paramsPerLayer} />
                <SidebarItem isDark label="Total Params" value={currentModel.specs.totalParams} />
                <SidebarItem isDark label="Est. VRAM (Full Fine-tune)" value={currentModel.specs.vramTraining} />
              </SidebarSection>

              <SidebarSection title="Parameter Breakdown" isDark>
                {currentModel.paramBreakdown.map((item, i) => (
                  <InfoItem key={i} isDark label={item.label} value={item.value} detail={item.detail} />
                ))}
              </SidebarSection>

              <SidebarSection title="VRAM Breakdown" isDark>
                {currentModel.memoryBreakdown.map((item, i) => (
                  <InfoItem key={i} isDark label={item.label} value={item.value} detail={item.detail} />
                ))}
              </SidebarSection>

              <SidebarSection title="Architecture Notes" isDark>
                <ul className="space-y-2">
                  {currentModel.architectureNotes.map((note, i) => (
                    <li key={i} className="flex gap-2 text-xs leading-relaxed" style={{ color: '#9c9189' }}>
                      <span style={{ color: '#c9994e', flexShrink: 0 }}>—</span>
                      <span>{note}</span>
                    </li>
                  ))}
                </ul>
              </SidebarSection>
            </motion.div>
          </div>

          <div className="p-4 text-[10px] text-center" style={{ borderTop: '1px solid #332e2a', color: '#6b6057' }}>
            Interactive WebGL Visualization • React Three Fiber
          </div>
        </div>

        <div className="flex-1 relative cursor-move">
          <Visualizer3D model={currentModel} activeStageId={activeStage?.id || null} onStageSelect={setActiveStage} isDark={true} />
          <MathPanel stage={activeStage} isDark={true} />
        </div>
      </div>
    );
  }

  // ── Light mode: original clean slate palette ──
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 text-slate-800">
      <div className="w-80 md:w-96 flex-shrink-0 bg-white border-r border-slate-200 flex flex-col z-20 shadow-lg">

        <div className="p-6 border-b border-slate-100 flex items-start justify-between">
          <div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-700 to-slate-500">
              LayerGraph
            </h1>
            <p className="text-xs text-slate-400 mt-1">Deep Learning Architecture Explorer</p>
          </div>
          <button onClick={() => setIsDark(true)} className="mt-0.5 p-2 rounded-lg bg-slate-100 text-slate-500 border border-slate-200 hover:bg-slate-200 transition-colors" title="Switch to dark mode">
            <MoonIcon />
          </button>
        </div>

        <div className="p-4 bg-slate-50/50">
          <div className="flex bg-slate-200 p-1 rounded-lg gap-1">
            {([
              [ModelType.TRANSFORMER_7B, '7B Transf.'],
              [ModelType.PERSONA_PLEX, 'PersonaPlex'],
              [ModelType.WAV2VEC_100M, 'Wav2Vec 2'],
            ] as [ModelType, string][]).map(([type, label]) => (
              <button key={type} onClick={() => setSelectedModelType(type)}
                className={`flex-1 py-2 text-[10px] md:text-xs font-semibold rounded-md transition-all ${
                  selectedModelType === type ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                }`}>
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 scrollbar-hide">
          <motion.div key={selectedModelType} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.3 }}>
            <div className="mb-5">
              <h2 className="text-lg font-semibold text-slate-800 mb-2">{currentModel.name}</h2>
              <p className="text-sm text-slate-500 leading-relaxed">{currentModel.description}</p>
            </div>

            <SidebarSection title="Training Specs" isDark={false}>
              <SidebarItem isDark={false} label="Hidden Size (H)" value={currentModel.specs.hiddenSize} />
              <SidebarItem isDark={false} label="Layers (L)" value={currentModel.specs.layers} />
              <SidebarItem isDark={false} label="Attention Heads" value={currentModel.specs.heads} sub={`Head Dim: ${currentModel.specs.hiddenSize / currentModel.specs.heads}`} />
              <SidebarItem isDark={false} label="FFN / MLP Size" value={currentModel.specs.ffnSize} />
              {currentModel.specs.convLayers && <SidebarItem isDark={false} label="CNN Layers" value={currentModel.specs.convLayers} />}
            </SidebarSection>

            <SidebarSection title="Memory & Params" isDark={false}>
              <SidebarItem isDark={false} label="Params / Layer" value={currentModel.specs.paramsPerLayer} />
              <SidebarItem isDark={false} label="Total Params" value={currentModel.specs.totalParams} />
              <SidebarItem isDark={false} label="Est. VRAM (Full Fine-tune)" value={currentModel.specs.vramTraining} />
            </SidebarSection>

            <SidebarSection title="Parameter Breakdown" isDark={false}>
              {currentModel.paramBreakdown.map((item, i) => (
                <InfoItem key={i} isDark={false} label={item.label} value={item.value} detail={item.detail} />
              ))}
            </SidebarSection>

            <SidebarSection title="VRAM Breakdown" isDark={false}>
              {currentModel.memoryBreakdown.map((item, i) => (
                <InfoItem key={i} isDark={false} label={item.label} value={item.value} detail={item.detail} />
              ))}
            </SidebarSection>

            <SidebarSection title="Architecture Notes" isDark={false}>
              <ul className="space-y-2">
                {currentModel.architectureNotes.map((note, i) => (
                  <li key={i} className="flex gap-2 text-xs text-slate-500 leading-relaxed">
                    <span className="text-slate-300 flex-shrink-0">—</span>
                    <span>{note}</span>
                  </li>
                ))}
              </ul>
            </SidebarSection>
          </motion.div>
        </div>

        <div className="p-4 border-t border-slate-100 text-[10px] text-slate-400 text-center">
          Interactive WebGL Visualization • React Three Fiber
        </div>
      </div>

      <div className="flex-1 relative cursor-move">
        <Visualizer3D model={currentModel} activeStageId={activeStage?.id || null} onStageSelect={setActiveStage} isDark={false} />
        <MathPanel stage={activeStage} isDark={false} />
      </div>
    </div>
  );
};

export default App;
