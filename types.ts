
export enum ModelType {
  TRANSFORMER_7B = 'TRANSFORMER_7B',
  WAV2VEC_100M = 'WAV2VEC_100M',
  PERSONA_PLEX = 'PERSONA_PLEX',
}

export interface MathFormula {
  label: string;
  latex: string;
  description: string;
  paramCount?: string;
  shapeCheck?: string;
}

export type StageCategory = 'data' | 'operation' | 'summary';

export interface ProcessingStage {
  id: string;
  title: string;
  type: 'input' | 'attention' | 'mlp' | 'norm' | 'cnn' | 'output';
  category: StageCategory;
  description: string;
  formulas: MathFormula[];
  position: [number, number, number];
  dimensions: [number, number, number];
  color: string;
  dimLabel: string;
  group?: string;
}

export interface ModelInfoItem {
  label: string;
  value: string;
  detail?: string; // smaller third line, e.g. formula or footnote
}

export interface ModelConfig {
  id: ModelType;
  name: string;
  description: string;
  specs: {
    hiddenSize: number;
    layers: number;
    heads: number;
    ffnSize: number;
    vocabSize?: number;
    convLayers?: number;
    paramsPerLayer: string;
    totalParams: string;
    vramTraining: string;
  };
  paramBreakdown: ModelInfoItem[];
  memoryBreakdown: ModelInfoItem[];
  architectureNotes: string[];
  stages: ProcessingStage[];
}
