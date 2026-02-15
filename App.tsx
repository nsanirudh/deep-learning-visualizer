
import React, { useState } from 'react';
import { ModelType, ProcessingStage } from './types';
import { MODELS } from './constants';
import { Visualizer3D } from './components/Visualizer3D';
import { MathPanel } from './components/MathPanel';
import { motion } from 'framer-motion';

const SidebarItem = ({ 
  label, 
  value, 
  sub 
}: { 
  label: string; 
  value: string | number; 
  sub?: string 
}) => (
  <div className="flex flex-col py-2 border-b border-slate-100 last:border-0">
    <span className="text-xs text-slate-400 uppercase tracking-wider mb-0.5">{label}</span>
    <span className="text-sm font-medium text-slate-700 font-mono">{value}</span>
    {sub && <span className="text-[10px] text-slate-400">{sub}</span>}
  </div>
);

const App = () => {
  const [selectedModelType, setSelectedModelType] = useState<ModelType>(ModelType.TRANSFORMER_7B);
  const [activeStage, setActiveStage] = useState<ProcessingStage | null>(null);

  const currentModel = MODELS[selectedModelType];

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 text-slate-800">
      
      {/* Sidebar Panel */}
      <div className="w-80 md:w-96 flex-shrink-0 bg-white border-r border-slate-200 flex flex-col z-20 shadow-lg">
        {/* Header */}
        <div className="p-6 border-b border-slate-100">
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-700 to-slate-500">
            NeuroVis 3D
          </h1>
          <p className="text-xs text-slate-400 mt-1">Deep Learning Architecture Explorer</p>
        </div>

        {/* Model Selector */}
        <div className="p-4 bg-slate-50/50">
          <div className="flex bg-slate-200 p-1 rounded-lg gap-1">
            <button
              onClick={() => setSelectedModelType(ModelType.TRANSFORMER_7B)}
              className={`flex-1 py-2 text-[10px] md:text-xs font-semibold rounded-md transition-all ${
                selectedModelType === ModelType.TRANSFORMER_7B
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              7B Transf.
            </button>
            <button
              onClick={() => setSelectedModelType(ModelType.PERSONA_PLEX)}
              className={`flex-1 py-2 text-[10px] md:text-xs font-semibold rounded-md transition-all ${
                selectedModelType === ModelType.PERSONA_PLEX
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              PersonaPlex
            </button>
            <button
              onClick={() => setSelectedModelType(ModelType.WAV2VEC_100M)}
              className={`flex-1 py-2 text-[10px] md:text-xs font-semibold rounded-md transition-all ${
                selectedModelType === ModelType.WAV2VEC_100M
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              Wav2Vec 2
            </button>
          </div>
        </div>

        {/* Model Stats */}
        <div className="flex-1 overflow-y-auto p-6 scrollbar-hide">
          <motion.div
            key={selectedModelType}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="mb-6">
              <h2 className="text-lg font-semibold text-slate-800 mb-2">{currentModel.name}</h2>
              <p className="text-sm text-slate-500 leading-relaxed">
                {currentModel.description}
              </p>
            </div>

            <div className="bg-slate-50 rounded-xl p-4 border border-slate-100 mb-6">
              <h3 className="text-xs font-bold text-slate-800 uppercase mb-3">Training Specs</h3>
              <SidebarItem label="Hidden Size (H)" value={currentModel.specs.hiddenSize} />
              <SidebarItem label="Layers (L)" value={currentModel.specs.layers} />
              <SidebarItem label="Attention Heads" value={currentModel.specs.heads} sub={`Head Dim: ${currentModel.specs.hiddenSize / currentModel.specs.heads}`} />
              <SidebarItem label="FFN / MLP Size" value={currentModel.specs.ffnSize} />
              {currentModel.specs.convLayers && (
                <SidebarItem label="CNN Layers" value={currentModel.specs.convLayers} />
              )}
            </div>

             <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
              <h3 className="text-xs font-bold text-slate-800 uppercase mb-3">Memory & Params</h3>
              <SidebarItem label="Params / Layer" value={currentModel.specs.paramsPerLayer} />
              <SidebarItem label="Total Params" value={currentModel.specs.totalParams} />
              <SidebarItem label="Est. VRAM (Full Fine-tune)" value={currentModel.specs.vramTraining} />
            </div>
          </motion.div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-100 text-[10px] text-slate-400 text-center">
          Interactive WebGL Visualization • React Three Fiber
        </div>
      </div>

      {/* Main 3D Viewport */}
      <div className="flex-1 relative">
        <Visualizer3D 
          model={currentModel} 
          activeStageId={activeStage?.id || null} 
          onStageSelect={setActiveStage}
        />
        
        {/* Math & Detail Overlay */}
        <MathPanel stage={activeStage} />
      </div>
    </div>
  );
};

export default App;
