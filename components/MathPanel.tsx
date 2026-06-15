import React, { useMemo } from 'react';
import { ProcessingStage, MathFormula } from '../types';
import { motion, AnimatePresence } from 'framer-motion';
import katex from 'katex';

interface MathPanelProps {
  stage: ProcessingStage | null;
  isDark: boolean;
  batchSize?: number;
  seqLength?: number;
}

const FormulaItem: React.FC<{ formula: MathFormula; isDark: boolean }> = ({ formula, isDark }) => {
  const html = useMemo(() => {
    try {
      return katex.renderToString(formula.latex, { throwOnError: false, displayMode: true });
    } catch (e) {
      return formula.latex;
    }
  }, [formula.latex]);

  if (!isDark) {
    return (
      <div className="mb-4 last:mb-0 p-3 bg-slate-50 rounded-lg border border-slate-100">
        <div className="flex justify-between items-baseline mb-1">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">{formula.label}</span>
          <span className="text-[10px] text-slate-400 italic font-mono">{formula.description}</span>
        </div>
        <div
          className="text-slate-800 text-sm md:text-base bg-white p-2 rounded border border-slate-200 shadow-sm overflow-x-auto min-h-[3rem] flex items-center justify-center"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>
    );
  }

  return (
    <div className="mb-4 last:mb-0 p-3 rounded-lg" style={{ background: '#1c1916', border: '1px solid #332e2a' }}>
      <div className="flex justify-between items-baseline mb-1">
        <span className="text-xs font-bold uppercase tracking-wider" style={{ color: '#c9994e' }}>{formula.label}</span>
        <span className="text-[10px] italic font-mono" style={{ color: '#6b6057' }}>{formula.description}</span>
      </div>
      <div
        className="text-sm md:text-base p-2 rounded overflow-x-auto min-h-[3rem] flex items-center justify-center"
        style={{ background: '#252220', border: '1px solid #3a3530', color: '#f0ebe4' }}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
};

// Dynamic mapping of actual evaluated shape details in the active stage
const getEvaluatedDetails = (stageId: string, B: number, S: number) => {
  const formattedElements = (num: number) => {
    return num >= 1e6 ? `${(num / 1e6).toFixed(2)}M` : num.toLocaleString();
  };

  switch (stageId) {
    case 'input':
      return {
        shape: `(${B}, ${S}, 4096)`,
        elements: B * S * 4096,
        detail: 'Initial vocabulary tokens encoded into dense float representations.'
      };
    case 'ln1_op':
    case 'ln1_out':
      return {
        shape: `(${B}, ${S}, 4096)`,
        elements: B * S * 4096,
        detail: 'Pre-Attention normalized tensor keeping variance scaled across activations.'
      };
    case 'q_proj':
    case 'k_proj':
    case 'v_proj':
      return {
        shape: `(4096, 4096)`,
        elements: 4096 * 4096,
        detail: 'Dense weight parameters projection matrix containing frozen structural coefficients.'
      };
    case 'q_tensor':
    case 'k_tensor':
    case 'v_tensor':
      return {
        shape: `(${B}, 32, ${S}, 128)`,
        elements: B * S * 4096,
        detail: 'Query, Key, or Value states reshaped for parallel Attention Heads (Head Dim = 128).'
      };
    case 'attn_op':
      return {
        shape: `(${B}, 32, ${S}, ${S})`,
        elements: B * 32 * S * S,
        detail: `Unified attention coefficient maps. Requires ${(B * 32 * S * S * 4 / 1e6).toFixed(1)} MB state memory!`
      };
    case 'attn_out_tensor':
    case 'wo_proj':
    case 'wo_out':
    case 'res_add_1':
      return {
        shape: `(${B}, ${S}, 4096)`,
        elements: B * S * 4096,
        detail: 'Concatenated projection output with linear weights in standard sequence stream.'
      };
    case 'ln2_op':
    case 'ln2_out':
      return {
        shape: `(${B}, ${S}, 4096)`,
        elements: B * S * 4096,
        detail: 'Normalized sequence activations ready for high-dimensional expansion.'
      };
    case 'w_gate_up':
      return {
        shape: `(4096, 11008)`,
        elements: 4096 * 11008,
        detail: 'SwiGLU combined Gating and Up-scaling parameters.'
      };
    case 'mlp_hidden':
      return {
        shape: `(${B}, ${S}, 11008)`,
        elements: B * S * 11008,
        detail: 'Expanded wide representation of sequential data in hidden layer dimension.'
      };
    case 'w_down':
      return {
        shape: `(11008, 4096)`,
        elements: 11008 * 4096,
        detail: 'Down projection matrix compression back down to residual hidden dimension.'
      };
    case 'mlp_out':
    case 'res_add_2':
    case 'layer_out':
      return {
        shape: `(${B}, ${S}, 4096)`,
        elements: B * S * 4096,
        detail: 'Consolidated sequence representation vector for Layer 1 passed to succeeding blocks.'
      };

    // Wav2Vec 2.0 Base (95M params)
    case 'input_raw':
      return {
        shape: `(${B}, 1, ${S * 320})`,
        elements: B * S * 320,
        detail: 'Continuous raw audio waveform array values.'
      };
    case 'cnn_layer1':
      return {
        shape: `(512, 1, 10)`,
        elements: 512 * 1 * 10,
        detail: '1D Convolutional filter banks matching temporal frequencies.'
      };
    case 'cnn_feat_1':
      return {
        shape: `(${B}, 512, ${Math.floor(S * 64)})`,
        elements: B * 512 * Math.floor(S * 64),
        detail: 'Downsampled audial signal activation map.'
      };
    case 'cnn_stack':
      return {
        shape: `(6, 512, 512, 3)`,
        elements: 6 * 512 * 512 * 3,
        detail: 'Stacked convolutional filter banks downsampling signal sequences recursively.'
      };
    case 'cnn_out':
      return {
        shape: `(${B}, 512, ${S})`,
        elements: B * 512 * S,
        detail: 'Consolidated audio latent vector mappings.'
      };
    case 'transpose':
      return {
        shape: `(${B}, 512, ${S}) → (${B}, ${S}, 512)`,
        elements: B * S * 512,
        detail: 'Axes transposition matching transformer sequences.'
      };
    case 'transposed_feat':
      return {
        shape: `(${B}, ${S}, 512)`,
        elements: B * S * 512,
        detail: 'Audial sequence representation arranged in sequence-first dimensions.'
      };
    case 'feat_proj':
      return {
        shape: `(512, 768)`,
        elements: 512 * 768,
        detail: 'Linear bridge matching latent features to standard hidden size dimensions.'
      };
    case 'trans_input':
      return {
        shape: `(${B}, ${S}, 768)`,
        elements: B * S * 768,
        detail: 'Transformer input latents matching attention heads sequence specifications.'
      };
    case 'pos_conv':
      return {
        shape: `(768, 768, 128)`,
        elements: 768 * 768 * 128,
        detail: 'Positional relative CNN filter banks compiling structural spatial coordinates.'
      };
    case 'layer1_out':
    case 'layers_n':
      return {
        shape: `(${B}, ${S}, 768)`,
        elements: B * S * 768,
        detail: 'Latent sequence vector output representing contextual phonemes.'
      };
    case 'ctc_head':
      return {
        shape: `(768, 32)`,
        elements: 768 * 32,
        detail: 'Classification weights projecting sequence output nodes to character probabilities.'
      };

    // PersonaPlex Speech-to-Speech
    case 'input_waveform':
      return {
        shape: `(${B}, ${S * 1920})`,
        elements: B * S * 1920,
        detail: 'Continuous voice sequence at 24kHz recorded frames.'
      };
    case 'mimi_encoder':
      return {
        shape: `(512, 1, 10)`,
        elements: 512 * 1 * 10,
        detail: 'CNN vocoder converter translating audio waves to discrete features.'
      };
    case 'mimi_latent':
      return {
        shape: `(${B}, ${S}, 512)`,
        elements: B * S * 512,
        detail: 'Speech acoustic continuous latents mapped chronologically.'
      };
    case 'rvq_quant':
      return {
        shape: `(8, 2048, 256)`,
        elements: 8 * 2048 * 256,
        detail: 'Vector Quantizer codebook mappings.'
      };
    case 'audio_tokens':
    case 'audio_tokens_out':
      return {
        shape: `(${B}, ${S}, 8)`,
        elements: B * S * 8,
        detail: 'Acoustic indices sequence composed of 8 levels of codebook quantization values.'
      };
    case 'embedding_lookup':
      return {
        shape: `(8, 2048, 4096)`,
        elements: 8 * 2048 * 4096,
        detail: 'Speech parameters lookup tables projecting RVQs to backbone dimension size.'
      };
    case 'hybrid_prompt':
      return {
        shape: `(${B}, ${S + 512}, 4096)`,
        elements: B * (S + 512) * 4096,
        detail: 'Concatenated and packed audio embeddings, style context, and query text.'
      };
    case 'temp_trans_block':
    case 'temp_out':
      return {
        shape: `(${B}, ${S}, 4096)`,
        elements: B * S * 4096,
        detail: 'Unified chronological representation across temporal axes.'
      };
    case 'text_head':
      return {
        shape: `(4096, 32000)`,
        elements: 4096 * 32000,
        detail: 'Output text representation logits matching vocab classifications.'
      };
    case 'depth_trans':
      return {
        shape: `(${B}, 8, 1024)`,
        elements: B * 8 * 1024,
        detail: 'Autoregressive speech code prediction block representing depth representations.'
      };
    case 'rvq_decode':
      return {
        shape: `(${B}, ${S}, 512)`,
        elements: B * S * 512,
        detail: 'Discrete indexes translated to sum latent vectors.'
      };
    case 'mimi_decoder':
      return {
        shape: `(1, 512, 10)`,
        elements: 1 * 512 * 10,
        detail: 'Inverted convolutional layers synthesising acoustic signals.'
      };
    case 'audio_out':
      return {
        shape: `(${B}, ${S * 1920})`,
        elements: B * S * 1920,
        detail: 'Synthesised output audial voice waveform.'
      };

    default:
      return null;
  }
};

export const MathPanel: React.FC<MathPanelProps> = ({ stage, isDark, batchSize = 1, seqLength = 2048 }) => {
  const evalDetails = useMemo(() => {
    if (!stage) return null;
    return getEvaluatedDetails(stage.id, batchSize, seqLength);
  }, [stage, batchSize, seqLength]);

  const formattedElements = (num: number) => {
    return num >= 1e6 ? `${(num / 1e6).toFixed(2)} Million` : num.toLocaleString();
  };

  if (!isDark) {
    // Light mode panel
    return (
      <div className="absolute bottom-6 right-6 w-full max-w-md z-10 pointer-events-none">
        <AnimatePresence mode="wait">
          {stage ? (
            <motion.div key={stage.id} initial={{ opacity: 0, y: 15, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.97 }} transition={{ duration: 0.22 }}
              className="bg-white/95 backdrop-blur-md rounded-xl shadow-xl border border-slate-200 overflow-hidden pointer-events-auto">
              <div className="p-4 border-b border-slate-100 flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-slate-800 text-sm md:text-base">{stage.title}</h3>
                  <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold">{stage.type}</p>
                </div>
                <div className="w-4.5 h-4.5 rounded-full shadow-inner border border-white" style={{ backgroundColor: stage.color }} />
              </div>
              <div className="p-4 max-h-[60vh] overflow-y-auto">
                <p className="text-xs md:text-sm text-slate-600 mb-4 leading-relaxed">{stage.description}</p>
                {stage.formulas.length > 0 && (
                  <div className="space-y-2 mb-4">
                    {stage.formulas.map((f, i) => <FormulaItem key={i} formula={f} isDark={false} />)}
                  </div>
                )}

                {/* Live evaluated dimension and matrix details */}
                {evalDetails && (
                  <div className="p-3.5 bg-slate-50 border border-slate-200/60 rounded-lg shadow-sm">
                    <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2">Evaluated Shape (This Step)</h4>
                    <div className="flex gap-4 mb-2">
                      <div>
                        <span className="block text-[10px] text-slate-400 uppercase tracking-wider">Shape Matrix</span>
                        <span className="text-xs md:text-sm font-bold font-mono text-slate-800">{evalDetails.shape}</span>
                      </div>
                      <div className="border-l border-slate-200 pl-4">
                        <span className="block text-[10px] text-slate-400 uppercase tracking-wider">Act. Floating Nodes</span>
                        <span className="text-xs md:text-sm font-bold font-mono text-slate-800">{formattedElements(evalDetails.elements)}</span>
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-500 leading-relaxed font-sans">{evalDetails.detail}</p>
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="bg-white/85 backdrop-blur p-4 rounded-xl shadow border border-slate-200 text-slate-400 text-xs text-center uppercase tracking-wider font-semibold pointer-events-auto">
              Hover over blocks to activate dimension evaluations
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  }

  // Dark mode panel
  return (
    <div className="absolute bottom-6 right-6 w-full max-w-md z-10 pointer-events-none">
      <AnimatePresence mode="wait">
        {stage ? (
          <motion.div key={stage.id} initial={{ opacity: 0, y: 15, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.97 }} transition={{ duration: 0.22 }}
            className="backdrop-blur-md rounded-xl shadow-xl overflow-hidden pointer-events-auto"
            style={{ background: 'rgba(28,25,22,0.96)', border: '1px solid #332e2a' }}>
            <div className="p-4 flex items-center justify-between" style={{ borderBottom: '1px solid #332e2a' }}>
              <div>
                <h3 className="font-semibold text-sm md:text-base" style={{ color: '#f0ebe4' }}>{stage.title}</h3>
                <p className="text-[10px] uppercase tracking-widest font-bold" style={{ color: '#6b6057' }}>{stage.type}</p>
              </div>
              <div className="w-4.5 h-4.5 rounded-full shadow-inner border border-[#151210]" style={{ backgroundColor: stage.color }} />
            </div>
            <div className="p-4 max-h-[60vh] overflow-y-auto">
              <p className="text-xs md:text-sm mb-4 leading-relaxed" style={{ color: '#9c9189' }}>{stage.description}</p>
              {stage.formulas.length > 0 && (
                <div className="space-y-2 mb-4">
                  {stage.formulas.map((f, i) => <FormulaItem key={i} formula={f} isDark={true} />)}
                </div>
              )}

              {/* Live evaluated dimension and matrix details */}
              {evalDetails && (
                <div className="p-3.5 rounded-lg border shadow-inner" style={{ background: '#1c1916', borderColor: '#332e2a' }}>
                  <h4 className="text-[10px] font-bold uppercase tracking-wider mb-2" style={{ color: '#6b6057' }}>Evaluated Shape (This Step)</h4>
                  <div className="flex gap-4 mb-2">
                    <div>
                      <span className="block text-[10px] uppercase tracking-wider" style={{ color: '#6b6057' }}>Shape Matrix</span>
                      <span className="text-xs md:text-sm font-bold font-mono" style={{ color: '#d4a85a' }}>{evalDetails.shape}</span>
                    </div>
                    <div className="border-l pl-4" style={{ borderColor: '#332e2a' }}>
                      <span className="block text-[10px] uppercase tracking-wider" style={{ color: '#6b6057' }}>Active Nodes</span>
                      <span className="text-xs md:text-sm font-bold font-mono" style={{ color: '#f0ebe4' }}>{formattedElements(evalDetails.elements)}</span>
                    </div>
                  </div>
                  <p className="text-[11px] leading-relaxed font-sans" style={{ color: '#9c9189' }}>{evalDetails.detail}</p>
                </div>
              )}
            </div>
          </motion.div>
        ) : (
          <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="backdrop-blur p-4 rounded-xl shadow text-xs text-center uppercase tracking-wider font-semibold pointer-events-auto border"
            style={{ background: 'rgba(28,25,22,0.85)', borderColor: '#332e2a', color: '#6b6057' }}>
            Hover over blocks to activate dimension evaluations
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
