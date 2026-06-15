import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import { motion, AnimatePresence } from 'framer-motion';
import { ModelType } from '../types';
import './notes.css';

import transformerNotes from '../data/transformer-7b.md?raw';
import wav2vecNotes from '../data/wav2vec.md?raw';
import personaNotes from '../data/persona-plex.md?raw';

const NOTES: Record<string, string> = {
  [ModelType.TRANSFORMER_7B]: transformerNotes,
  [ModelType.WAV2VEC_100M]: wav2vecNotes,
  [ModelType.PERSONA_PLEX]: personaNotes,
};

const MODEL_LABELS: Record<string, string> = {
  [ModelType.TRANSFORMER_7B]: 'LLaMA 7B Decoder',
  [ModelType.WAV2VEC_100M]: 'Wav2Vec 2.0 · 95M',
  [ModelType.PERSONA_PLEX]: 'PersonaPlex · 7.2B',
};

const MODEL_TAGS: Record<string, string> = {
  [ModelType.TRANSFORMER_7B]: 'LLM DECODER',
  [ModelType.WAV2VEC_100M]: 'SPEECH LEARNER',
  [ModelType.PERSONA_PLEX]: 'SPEECH-TO-SPEECH',
};

interface NotesPanelProps {
  modelType: ModelType;
  isDark: boolean;
  isOpen: boolean;
  onClose: () => void;
}

const PANEL_WIDTH = 560;

export const NotesPanel: React.FC<NotesPanelProps> = ({ modelType, isDark, isOpen, onClose }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && scrollRef.current) scrollRef.current.scrollTop = 0;
  }, [isOpen, modelType]);

  const notes = NOTES[modelType] ?? '';

  const cssVarsDark = {
    '--np-border': '#2a3252',
    '--np-code-bg': '#0c0e16',
    '--np-stripe': 'rgba(255,255,255,0.02)',
    '--np-accent': '#818cf8',
    '--np-text-primary': '#e2e8f0',
    '--np-text-secondary': '#94a3b8',
    '--np-text-dim': '#6a7ba2',
  } as React.CSSProperties;

  const cssVarsLight = {
    '--np-border': '#e2e8f0',
    '--np-code-bg': '#f8fafc',
    '--np-stripe': 'rgba(0,0,0,0.02)',
    '--np-accent': '#4f46e5',
    '--np-text-primary': '#0f172a',
    '--np-text-secondary': '#334155',
    '--np-text-dim': '#64748b',
  } as React.CSSProperties;

  const vars = isDark ? cssVarsDark : cssVarsLight;

  return (
    <AnimatePresence>
      {isOpen && (
        // Outer: animates the width so the canvas shrinks smoothly
        <motion.div
          key="notes-outer"
          initial={{ width: 0 }}
          animate={{ width: PANEL_WIDTH }}
          exit={{ width: 0 }}
          transition={{ type: 'spring', damping: 32, stiffness: 340, mass: 0.9 }}
          style={{
            flexShrink: 0,
            overflow: 'hidden',
            height: '100%',
            borderLeft: `1px solid ${isDark ? '#2a3252' : '#e2e8f0'}`,
          }}
        >
          {/* Inner: fixed width so content doesn't reflow during animation */}
          <div
            style={{
              width: PANEL_WIDTH,
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              background: isDark ? '#0d0f1c' : '#ffffff',
              ...vars,
            }}
          >
            {/* ── Header ── */}
            <div
              style={{
                flexShrink: 0,
                padding: '14px 20px 12px',
                borderBottom: `1px solid ${isDark ? '#2a3252' : '#e2e8f0'}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: isDark ? '#0c0e16' : '#f8fafc',
              }}
            >
              <div>
                <span style={{
                  display: 'inline-block',
                  fontSize: '9px',
                  fontWeight: 800,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  color: isDark ? '#6a7ba2' : '#94a3b8',
                  marginBottom: '3px',
                }}>
                  {MODEL_TAGS[modelType]}
                </span>
                <div style={{
                  fontSize: '15px',
                  fontWeight: 700,
                  letterSpacing: '-0.01em',
                  color: isDark ? '#818cf8' : '#4f46e5',
                  lineHeight: 1.2,
                }}>
                  {MODEL_LABELS[modelType]}
                </div>
              </div>

              <button
                onClick={onClose}
                style={{
                  fontSize: '11px',
                  fontWeight: 700,
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                  padding: '5px 10px',
                  borderRadius: '7px',
                  border: `1px solid ${isDark ? '#2a3252' : '#e2e8f0'}`,
                  background: isDark ? '#1a1d2e' : '#f1f5f9',
                  color: isDark ? '#6a7ba2' : '#64748b',
                  cursor: 'pointer',
                  transition: 'opacity 0.15s',
                }}
              >
                ✕
              </button>
            </div>

            {/* ── Scrollable markdown body ── */}
            <div
              ref={scrollRef}
              style={{
                flex: 1,
                overflowY: 'auto',
                padding: '24px 28px 40px',
              }}
            >
              <div
                className="notes-prose"
                style={{ color: isDark ? '#e2e8f0' : '#0f172a' }}
              >
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {notes}
                </ReactMarkdown>
              </div>
            </div>

            {/* ── Footer ── */}
            <div style={{
              flexShrink: 0,
              padding: '8px 20px',
              borderTop: `1px solid ${isDark ? '#2a3252' : '#e2e8f0'}`,
              fontSize: '9px',
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: isDark ? '#3b4270' : '#cbd5e1',
            }}>
              Personal deep learning notes
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
