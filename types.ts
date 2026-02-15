
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
  position: [number, number, number]; // 3D position
  dimensions: [number, number, number]; // 3D size
  color: string;
  dimLabel: string; // Explicit dimension label to display on block
  group?: string; // ID for grouping (e.g. 'layer_1')
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
  stages: ProcessingStage[];
}
