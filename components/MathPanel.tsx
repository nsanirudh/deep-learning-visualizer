
import React, { useMemo } from 'react';
import { ProcessingStage, MathFormula } from '../types';
import { motion, AnimatePresence } from 'framer-motion';
import katex from 'katex';

interface MathPanelProps {
  stage: ProcessingStage | null;
}

const FormulaItem: React.FC<{ formula: MathFormula }> = ({ formula }) => {
  const html = useMemo(() => {
    try {
      return katex.renderToString(formula.latex, {
        throwOnError: false,
        displayMode: true,
      });
    } catch (e) {
      return formula.latex;
    }
  }, [formula.latex]);

  return (
    <div className="mb-4 last:mb-0 p-3 bg-slate-50 rounded-lg border border-slate-100">
      <div className="flex justify-between items-baseline mb-1">
        <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">{formula.label}</span>
        <span className="text-xs text-slate-400 italic font-mono">{formula.description}</span>
      </div>
      <div 
        className="text-slate-800 text-sm md:text-base bg-white p-2 rounded border border-slate-200 shadow-sm overflow-x-auto min-h-[3rem] flex items-center justify-center"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
};

export const MathPanel: React.FC<MathPanelProps> = ({ stage }) => {
  return (
    <div className="absolute bottom-6 right-6 w-full max-w-md z-10 pointer-events-none">
       <AnimatePresence mode="wait">
        {stage ? (
          <motion.div
            key={stage.id}
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.3 }}
            className="bg-white/90 backdrop-blur-md rounded-xl shadow-xl border border-slate-200 overflow-hidden pointer-events-auto"
          >
            <div className="p-4 border-b border-slate-100 flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-slate-800">{stage.title}</h3>
                <p className="text-xs text-slate-500 uppercase tracking-widest">{stage.type}</p>
              </div>
              <div 
                className="w-4 h-4 rounded-full shadow-inner" 
                style={{ backgroundColor: stage.color }}
              />
            </div>
            
            <div className="p-4 max-h-[60vh] overflow-y-auto">
              <p className="text-sm text-slate-600 mb-4 leading-relaxed">
                {stage.description}
              </p>
              
              {stage.formulas.length > 0 && (
                <div className="space-y-2">
                  {stage.formulas.map((f, i) => (
                    <FormulaItem key={i} formula={f} />
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="bg-white/80 backdrop-blur p-4 rounded-xl shadow border border-slate-200 text-slate-500 text-sm text-center italic pointer-events-auto"
          >
            Hover over a layer block to see details.
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
