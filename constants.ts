

import { ModelConfig, ModelType, ProcessingStage } from './types';

// Color Palette
const C = {
  DATA: '#94A3B8', // Slate 400 (Grey for data tensors)
  INPUT: '#CBD5E1', // Slate 300
  ATTN: '#FBCFE8', // Pink 200
  MLP: '#BAE6FD', // Blue 200
  NORM: '#BBF7D0', // Green 200
  CNN: '#DDD6FE', // Violet 200
  OUTPUT: '#E2E8F0',
  SUMMARY: '#F1F5F9',
};

const DATA_HEIGHT = 0.2;
const OP_HEIGHT = 0.8;

export const MODELS: Record<ModelType, ModelConfig> = {
  [ModelType.TRANSFORMER_7B]: {
    id: ModelType.TRANSFORMER_7B,
    name: "7B Decoder Transformer",
    description: "A LLaMA-style Large Language Model. Showing detailed flow for Layer 1 with RMSNorm and SwiGLU.",
    specs: {
      hiddenSize: 4096,
      layers: 32,
      heads: 32,
      ffnSize: 11008,
      vocabSize: 32000,
      paramsPerLayer: "~202M",
      totalParams: "~6.47B (Blocks) / ~7B Total",
      vramTraining: "~112 GB (AdamW Mixed Precision)",
    },
    stages: [
      // --- INPUT ---
      {
        id: 'input',
        title: 'Input Embeddings',
        type: 'input',
        category: 'data',
        description: 'Initial token embeddings from vocabulary.',
        formulas: [
          { label: 'Lookup', latex: 'X = W_{emb}[tok]', description: 'Embedding Lookup' },
          { label: 'Shape', latex: '(B, S, H)', description: 'Batch, Seq, Hidden' }
        ],
        position: [0, 0, 0],
        dimensions: [4, DATA_HEIGHT, 4],
        color: C.DATA,
        dimLabel: '(B, S, 4096)'
      },
      
      // --- LAYER 1 START ---
      {
        id: 'ln1_op',
        title: 'RMS Norm',
        type: 'norm',
        category: 'operation',
        description: 'Root Mean Square Normalization.',
        formulas: [
          { label: 'Eq', latex: '\\frac{x}{\\sqrt{\\frac{1}{H}\\sum x_i^2 + \\epsilon}} * g', description: 'Scale invariant norm' },
          { label: 'Params', latex: 'g \\in \\mathbb{R}^H', description: 'Gain vector' }
        ],
        position: [0, 1.5, 0],
        dimensions: [3, OP_HEIGHT, 3],
        color: C.NORM,
        dimLabel: 'H',
        group: 'layer_1'
      },
      {
        id: 'ln1_out',
        title: 'Normed Input',
        type: 'norm',
        category: 'data',
        description: 'Normalized input ready for attention.',
        formulas: [],
        position: [0, 2.5, 0],
        dimensions: [4, DATA_HEIGHT, 4],
        color: C.DATA,
        dimLabel: '(B, S, 4096)',
        group: 'layer_1'
      },
      
      // QKV Projections
      {
        id: 'q_proj',
        title: 'W_q Matrix',
        type: 'attention',
        category: 'operation',
        description: 'Query Projection.',
        formulas: [{ label: 'Params', latex: 'H \\times H', description: '~16.7M' }],
        position: [-2.5, 4, 0],
        dimensions: [1.5, OP_HEIGHT, 1.5],
        color: C.ATTN,
        dimLabel: '4096 x 4096',
        group: 'layer_1'
      },
      {
        id: 'k_proj',
        title: 'W_k Matrix',
        type: 'attention',
        category: 'operation',
        description: 'Key Projection.',
        formulas: [{ label: 'Params', latex: 'H \\times H', description: '~16.7M' }],
        position: [0, 4, 0],
        dimensions: [1.5, OP_HEIGHT, 1.5],
        color: C.ATTN,
        dimLabel: '4096 x 4096',
        group: 'layer_1'
      },
      {
        id: 'v_proj',
        title: 'W_v Matrix',
        type: 'attention',
        category: 'operation',
        description: 'Value Projection.',
        formulas: [{ label: 'Params', latex: 'H \\times H', description: '~16.7M' }],
        position: [2.5, 4, 0],
        dimensions: [1.5, OP_HEIGHT, 1.5],
        color: C.ATTN,
        dimLabel: '4096 x 4096',
        group: 'layer_1'
      },
      
      // QKV Tensors
      {
        id: 'q_tensor',
        title: 'Query (Q)',
        type: 'attention',
        category: 'data',
        description: 'Projected Query Tensor.',
        formulas: [{ label: 'Q', latex: 'X W_q', description: '' }],
        position: [-2.5, 5, 0],
        dimensions: [1.5, DATA_HEIGHT, 1.5],
        color: C.DATA,
        dimLabel: '(B, S, 4096)',
        group: 'layer_1'
      },
      {
        id: 'k_tensor',
        title: 'Key (K)',
        type: 'attention',
        category: 'data',
        description: 'Projected Key Tensor.',
        formulas: [{ label: 'K', latex: 'X W_k', description: '' }],
        position: [0, 5, 0],
        dimensions: [1.5, DATA_HEIGHT, 1.5],
        color: C.DATA,
        dimLabel: '(B, S, 4096)',
        group: 'layer_1'
      },
      {
        id: 'v_tensor',
        title: 'Value (V)',
        type: 'attention',
        category: 'data',
        description: 'Projected Value Tensor.',
        formulas: [{ label: 'V', latex: 'X W_v', description: '' }],
        position: [2.5, 5, 0],
        dimensions: [1.5, DATA_HEIGHT, 1.5],
        color: C.DATA,
        dimLabel: '(B, S, 4096)',
        group: 'layer_1'
      },
      
      // Attention Operation
      {
        id: 'attn_op',
        title: 'Self Attention',
        type: 'attention',
        category: 'operation',
        description: 'Scaled Dot-Product Attention.',
        formulas: [
            { label: 'Func', latex: '\\text{softmax}(\\frac{QK^T}{\\sqrt{d}} + M)', description: 'Masked Attention' },
            { label: 'Score', latex: '(B, n_h, S, S)', description: 'Attn Weights' }
        ],
        position: [0, 6.5, 0],
        dimensions: [6.5, 0.5, 3],
        color: C.ATTN,
        dimLabel: 'Softmax(Score)',
        group: 'layer_1'
      },
      {
        id: 'attn_out_tensor',
        title: 'Context Vector (Z)',
        type: 'attention',
        category: 'data',
        description: 'Weighted sum of Values.',
        formulas: [{ label: 'Shape', latex: '(B, S, H)', description: 'Contextualized info' }],
        position: [0, 7.5, 0],
        dimensions: [4, DATA_HEIGHT, 4],
        color: C.DATA,
        dimLabel: '(B, S, 4096)',
        group: 'layer_1'
      },
      
      // Output Projection
      {
        id: 'wo_proj',
        title: 'W_o Matrix',
        type: 'attention',
        category: 'operation',
        description: 'Output Projection Weights.',
        formulas: [{ label: 'Params', latex: 'H \\times H', description: '~16.7M' }],
        position: [0, 8.5, 0],
        dimensions: [3, OP_HEIGHT, 3],
        color: C.ATTN,
        dimLabel: '4096 x 4096',
        group: 'layer_1'
      },
      {
        id: 'wo_out',
        title: 'Attn Output',
        type: 'attention',
        category: 'data',
        description: 'Output of Attention Block.',
        formulas: [],
        position: [0, 9.5, 0],
        dimensions: [4, DATA_HEIGHT, 4],
        color: C.DATA,
        dimLabel: '(B, S, 4096)',
        group: 'layer_1'
      },
      
      // Residual 1
      {
        id: 'res_add_1',
        title: 'Residual Add',
        type: 'output',
        category: 'operation',
        description: 'Element-wise addition.',
        formulas: [{ label: 'Op', latex: 'x + \\text{Attn}(x)', description: 'Skip connection' }],
        position: [0, 10.5, 0],
        dimensions: [4, 0.3, 4],
        color: C.OUTPUT,
        dimLabel: 'Add',
        group: 'layer_1'
      },
      
      // MLP Section
      {
        id: 'ln2_op',
        title: 'RMS Norm 2',
        type: 'norm',
        category: 'operation',
        description: 'Pre-MLP Normalization.',
        formulas: [{ label: 'Params', latex: 'H', description: 'Gain vector' }],
        position: [0, 12, 0],
        dimensions: [3, OP_HEIGHT, 3],
        color: C.NORM,
        dimLabel: 'H',
        group: 'layer_1'
      },
      {
        id: 'ln2_out',
        title: 'Normed Input 2',
        type: 'norm',
        category: 'data',
        description: 'Normalized residuals.',
        formulas: [],
        position: [0, 13, 0],
        dimensions: [4, DATA_HEIGHT, 4],
        color: C.DATA,
        dimLabel: '(B, S, 4096)',
        group: 'layer_1'
      },
      
      // SwiGLU Up/Gate
      {
        id: 'w_gate_up',
        title: 'Gate & Up (SwiGLU)',
        type: 'mlp',
        category: 'operation',
        description: 'Gated Linear Unit with Swish.',
        formulas: [
            { label: 'Params', latex: '2 \\times (H \\times I)', description: '~90M' },
            { label: 'Swish', latex: 'x \\cdot \\sigma(x)', description: 'Activation' },
            { label: 'Op', latex: '(XW_u) \\odot \\text{swish}(XW_g)', description: 'Element-wise mult' }
        ],
        position: [0, 14.5, 0],
        dimensions: [5, OP_HEIGHT, 5],
        color: C.MLP,
        dimLabel: '2 x (4096 x 11008)',
        group: 'layer_1'
      },
      {
        id: 'mlp_hidden',
        title: 'Hidden State',
        type: 'mlp',
        category: 'data',
        description: 'Intermediate wide representation.',
        formulas: [{ label: 'Shape', latex: '(B, S, I)', description: 'I = 11008' }],
        position: [0, 15.5, 0],
        dimensions: [5, DATA_HEIGHT, 5],
        color: C.DATA,
        dimLabel: '(B, S, 11008)',
        group: 'layer_1'
      },
      
      // Down Proj
      {
        id: 'w_down',
        title: 'Down Proj',
        type: 'mlp',
        category: 'operation',
        description: 'W_down matrix.',
        formulas: [{ label: 'Params', latex: 'I \\times H', description: '~45M' }],
        position: [0, 17, 0],
        dimensions: [4, OP_HEIGHT, 4],
        color: C.MLP,
        dimLabel: '11008 x 4096',
        group: 'layer_1'
      },
      {
        id: 'mlp_out',
        title: 'MLP Output',
        type: 'mlp',
        category: 'data',
        description: 'Projected back to hidden size.',
        formulas: [],
        position: [0, 18, 0],
        dimensions: [4, DATA_HEIGHT, 4],
        color: C.DATA,
        dimLabel: '(B, S, 4096)',
        group: 'layer_1'
      },
      
      // Residual 2
      {
        id: 'res_add_2',
        title: 'Residual Add',
        type: 'output',
        category: 'operation',
        description: 'Skip connection.',
        formulas: [{ label: 'Op', latex: 'x + \\text{MLP}(x)', description: '' }],
        position: [0, 19, 0],
        dimensions: [4, 0.3, 4],
        color: C.OUTPUT,
        dimLabel: 'Add',
        group: 'layer_1'
      },
      {
        id: 'layer_out',
        title: 'Layer 1 Output',
        type: 'output',
        category: 'data',
        description: 'Final output of Layer 1.',
        formulas: [],
        position: [0, 20, 0],
        dimensions: [4, DATA_HEIGHT, 4],
        color: C.DATA,
        dimLabel: '(B, S, 4096)',
        group: 'layer_1'
      },
      
      // --- SUMMARY ---
      {
        id: 'layers_n',
        title: 'Layers 2 - 32',
        type: 'output',
        category: 'summary',
        description: 'The block above is repeated 31 more times.',
        formulas: [{ label: 'Total Blocks', latex: '32', description: 'Deep network' }],
        position: [0, 22.5, 0],
        dimensions: [4, 3, 4],
        color: C.SUMMARY,
        dimLabel: 'x 31 Layers'
      }
    ]
  },
  
  [ModelType.WAV2VEC_100M]: {
    id: ModelType.WAV2VEC_100M,
    name: "Wav2Vec 2.0 Base (100M)",
    description: "Speech representation learner. 7-layer CNN Feature Encoder + 12-layer Transformer. Trained via Contrastive Loss or CTC.",
    specs: {
      hiddenSize: 768,
      layers: 12,
      heads: 8,
      ffnSize: 3072,
      convLayers: 7,
      vocabSize: 32,
      paramsPerLayer: "~7.1M",
      totalParams: "~95M",
      vramTraining: "~1.6 GB",
    },
    stages: [
      {
        id: 'input_raw',
        title: 'Raw Audio',
        type: 'input',
        category: 'data',
        description: 'Raw waveform float array.',
        formulas: [{ label: 'Shape', latex: 'x \\in \\mathbb{R}^{B \\times 1 \\times T}', description: 'Single Channel' }],
        position: [0, 0, 0],
        dimensions: [2, DATA_HEIGHT, 6],
        color: C.DATA,
        dimLabel: '(B, 1, T)'
      },
      // CNN Breakdown: Layer 1
      {
        id: 'cnn_layer1',
        title: 'CNN Layer 1',
        type: 'cnn',
        category: 'operation',
        description: 'High stride downsampling. Kernel=10, Stride=5.',
        formulas: [
          { label: 'Stride', latex: '5', description: 'Downsample x5' },
          { label: 'Channels', latex: '512', description: 'Feature maps' }
        ],
        position: [0, 1.2, 0],
        dimensions: [3, 0.8, 3],
        color: C.CNN,
        dimLabel: 'K=10, S=5'
      },
      {
        id: 'cnn_feat_1',
        title: 'Feature Map 1',
        type: 'cnn',
        category: 'data',
        description: 'Initial features.',
        formulas: [{ label: 'Time', latex: 'T/5', description: '' }],
        position: [0, 2.2, 0],
        dimensions: [3, DATA_HEIGHT, 4],
        color: C.DATA,
        dimLabel: '(B, 512, T/5)'
      },
      // CNN Breakdown: Layers 2-7
      {
        id: 'cnn_stack',
        title: 'CNN Layers 2-7',
        type: 'cnn',
        category: 'operation',
        description: 'Stack of 6 layers. Kernel=3, Stride=2 each.',
        formulas: [
          { label: 'Total Stride', latex: '2^6 = 64', description: 'Cumulative' },
          { label: 'Overall', latex: '320x Downsampling', description: '5 * 64' }
        ],
        position: [0, 3.4, 0],
        dimensions: [3, 1.5, 3],
        color: C.CNN,
        dimLabel: '6x (K=3, S=2)'
      },
      {
        id: 'cnn_out',
        title: 'Latent Features (z)',
        type: 'cnn',
        category: 'data',
        description: 'Final CNN output.',
        formulas: [
            { label: 'Output', latex: 'z \\in \\mathbb{R}^{B \\times 512 \\times S}', description: 'S = T/320' },
        ],
        position: [0, 4.8, 0],
        dimensions: [3, DATA_HEIGHT, 3],
        color: C.DATA,
        dimLabel: '(B, 512, S)'
      },
      // Transpose
      {
        id: 'transpose',
        title: 'Transpose',
        type: 'input',
        category: 'operation',
        description: 'Swap dimensions for Transformer.',
        formulas: [{ label: 'Op', latex: '(B, C, S) \\rightarrow (B, S, C)', description: '' }],
        position: [0, 5.6, 0],
        dimensions: [3, 0.4, 3],
        color: C.INPUT,
        dimLabel: 'Transpose'
      },
      {
        id: 'transposed_feat',
        title: 'Transposed Feats',
        type: 'input',
        category: 'data',
        description: 'Sequence-first format.',
        formulas: [{ label: 'Z', latex: 'Z \\in \\mathbb{R}^{B \\times S \\times 512}', description: '' }],
        position: [0, 6.4, 0],
        dimensions: [3.5, DATA_HEIGHT, 3.5],
        color: C.DATA,
        dimLabel: '(B, S, 512)'
      },
      // Projection
      {
        id: 'feat_proj',
        title: 'Feature Projector',
        type: 'input',
        category: 'operation',
        description: 'Linear projection to Transformer hidden size.',
        formulas: [
          { label: 'Proj', latex: 'X = Z W_{proj} + b', description: 'Bridge' },
          { label: 'Params', latex: '512 \\times 768', description: 'Dim Match' }
        ],
        position: [0, 7.4, 0],
        dimensions: [3.5, 0.8, 3.5],
        color: C.INPUT,
        dimLabel: '512 -> 768'
      },
      {
        id: 'trans_input',
        title: 'Transformer Input',
        type: 'input',
        category: 'data',
        description: 'Projected features.',
        formulas: [{ label: 'Shape', latex: '(B, S, 768)', description: 'Hidden size 768' }],
        position: [0, 8.4, 0],
        dimensions: [3.5, DATA_HEIGHT, 3.5],
        color: C.DATA,
        dimLabel: '(B, S, 768)',
        group: 'layer_1'
      },
      {
        id: 'pos_conv',
        title: 'Positional Conv',
        type: 'input',
        category: 'operation',
        description: 'Convolutional relative positioning.',
        formulas: [{ label: 'Add', latex: 'X \\leftarrow X + \\text{PosConv}(X)', description: 'Relative Pos' }],
        position: [0, 9.2, 0],
        dimensions: [3.5, 0.4, 3.5],
        color: C.NORM,
        dimLabel: 'Conv1D',
        group: 'layer_1'
      },
      
      // Layer Norm 1
       {
        id: 'ln1',
        title: 'Layer Norm',
        type: 'norm',
        category: 'operation',
        description: 'Standard Layer Normalization.',
        formulas: [{ label: 'Params', latex: '2H', description: 'Scale & Bias' }],
        position: [0, 10.0, 0],
        dimensions: [3, 0.5, 3],
        color: C.NORM,
        dimLabel: 'LN',
        group: 'layer_1'
      },

      // Attention Breakdown
      {
        id: 'q_proj',
        title: 'W_q Matrix',
        type: 'attention',
        category: 'operation',
        description: 'Query Projection (With Bias).',
        formulas: [{ label: 'Params', latex: '768 \\times 768', description: '+ Bias' }],
        position: [-2.5, 11.2, 0],
        dimensions: [1.5, 0.8, 1.5],
        color: C.ATTN,
        dimLabel: 'W_q',
        group: 'layer_1'
      },
      {
        id: 'k_proj',
        title: 'W_k Matrix',
        type: 'attention',
        category: 'operation',
        description: 'Key Projection (With Bias).',
        formulas: [{ label: 'Params', latex: '768 \\times 768', description: '+ Bias' }],
        position: [0, 11.2, 0],
        dimensions: [1.5, 0.8, 1.5],
        color: C.ATTN,
        dimLabel: 'W_k',
        group: 'layer_1'
      },
      {
        id: 'v_proj',
        title: 'W_v Matrix',
        type: 'attention',
        category: 'operation',
        description: 'Value Projection (With Bias).',
        formulas: [{ label: 'Params', latex: '768 \\times 768', description: '+ Bias' }],
        position: [2.5, 11.2, 0],
        dimensions: [1.5, 0.8, 1.5],
        color: C.ATTN,
        dimLabel: 'W_v',
        group: 'layer_1'
      },
      // QKV Tensors
      {
        id: 'q_tensor',
        title: 'Q',
        type: 'attention',
        category: 'data',
        description: 'Query Tensor.',
        formulas: [{ label: 'Shape', latex: '(B, 8, S, 96)', description: '8 Heads' }],
        position: [-2.5, 12.2, 0],
        dimensions: [1.5, DATA_HEIGHT, 1.5],
        color: C.DATA,
        dimLabel: 'Q',
        group: 'layer_1'
      },
      {
        id: 'k_tensor',
        title: 'K',
        type: 'attention',
        category: 'data',
        description: 'Key Tensor.',
        formulas: [{ label: 'Shape', latex: '(B, 8, S, 96)', description: '8 Heads' }],
        position: [0, 12.2, 0],
        dimensions: [1.5, DATA_HEIGHT, 1.5],
        color: C.DATA,
        dimLabel: 'K',
        group: 'layer_1'
      },
      {
        id: 'v_tensor',
        title: 'V',
        type: 'attention',
        category: 'data',
        description: 'Value Tensor.',
        formulas: [{ label: 'Shape', latex: '(B, 8, S, 96)', description: '8 Heads' }],
        position: [2.5, 12.2, 0],
        dimensions: [1.5, DATA_HEIGHT, 1.5],
        color: C.DATA,
        dimLabel: 'V',
        group: 'layer_1'
      },

      {
        id: 'attn_mix',
        title: 'Attention Mixing',
        type: 'attention',
        category: 'operation',
        description: 'Softmax and weighted sum.',
        formulas: [
            { label: 'Score', latex: 'A = QK^T / \\sqrt{96}', description: 'Scaled Dot Prod' },
            { label: 'Out', latex: 'Z = \\text{softmax}(A)V', description: 'Context' }
        ],
        position: [0, 13.5, 0],
        dimensions: [6.5, 0.5, 3],
        color: C.ATTN,
        dimLabel: 'Softmax(QK^T)',
        group: 'layer_1'
      },
      {
        id: 'attn_context',
        title: 'Context Vector',
        type: 'attention',
        category: 'data',
        description: 'Contextualized info.',
        formulas: [{ label: 'Z', latex: '(B, S, 768)', description: '' }],
        position: [0, 14.5, 0],
        dimensions: [3.5, DATA_HEIGHT, 3.5],
        color: C.DATA,
        dimLabel: '(B, S, 768)',
        group: 'layer_1'
      },
      {
        id: 'out_proj',
        title: 'Output Projection',
        type: 'attention',
        category: 'operation',
        description: 'Final Attention Linear Layer.',
        formulas: [{ label: 'W_o', latex: '768 \\times 768', description: '' }],
        position: [0, 15.3, 0],
        dimensions: [3, 0.6, 3],
        color: C.ATTN,
        dimLabel: 'W_o',
        group: 'layer_1'
      },
      {
        id: 'res_add_1',
        title: 'Residual Add',
        type: 'output',
        category: 'operation',
        description: 'Skip connection.',
        formulas: [],
        position: [0, 16.2, 0],
        dimensions: [3.5, 0.2, 3.5],
        color: C.OUTPUT,
        dimLabel: 'Add',
        group: 'layer_1'
      },
       {
        id: 'ln2',
        title: 'Layer Norm 2',
        type: 'norm',
        category: 'operation',
        description: 'Pre-FFN Norm.',
        formulas: [{ label: 'Params', latex: '2H', description: '' }],
        position: [0, 17.0, 0],
        dimensions: [3, 0.5, 3],
        color: C.NORM,
        dimLabel: 'LN',
        group: 'layer_1'
      },

      // FFN Breakdown
      {
        id: 'ffn_up',
        title: 'FFN Up (W1)',
        type: 'mlp',
        category: 'operation',
        description: 'Expansion Layer.',
        formulas: [{ label: 'Params', latex: '768 \\times 3072', description: '+ Bias' }],
        position: [0, 18.2, 0],
        dimensions: [4.5, 0.8, 4.5],
        color: C.MLP,
        dimLabel: 'W1',
        group: 'layer_1'
      },
      {
        id: 'ffn_act',
        title: 'Activation',
        type: 'mlp',
        category: 'operation',
        description: 'Non-linearity.',
        formulas: [{ label: 'Func', latex: '\\text{GELU}(x)', description: 'or ReLU' }],
        position: [0, 19.2, 0],
        dimensions: [4.5, 0.3, 4.5],
        color: C.MLP,
        dimLabel: 'Act',
        group: 'layer_1'
      },
      {
        id: 'ffn_mid',
        title: 'FFN Hidden',
        type: 'mlp',
        category: 'data',
        description: 'Intermediate activation.',
        formulas: [{ label: 'Shape', latex: '(B, S, 3072)', description: '' }],
        position: [0, 20.0, 0],
        dimensions: [4.5, DATA_HEIGHT, 4.5],
        color: C.DATA,
        dimLabel: '(B, S, 3072)',
        group: 'layer_1'
      },
      {
        id: 'ffn_down',
        title: 'FFN Down (W2)',
        type: 'mlp',
        category: 'operation',
        description: 'Projection back to H.',
        formulas: [{ label: 'Params', latex: '3072 \\times 768', description: '+ Bias' }],
        position: [0, 21.0, 0],
        dimensions: [3.5, 0.8, 3.5],
        color: C.MLP,
        dimLabel: 'W2',
        group: 'layer_1'
      },
      {
        id: 'res_add_2',
        title: 'Residual Add',
        type: 'output',
        category: 'operation',
        description: 'Skip connection.',
        formulas: [],
        position: [0, 22.0, 0],
        dimensions: [3.5, 0.2, 3.5],
        color: C.OUTPUT,
        dimLabel: 'Add',
        group: 'layer_1'
      },
      {
        id: 'layer1_out',
        title: 'Layer Output',
        type: 'output',
        category: 'data',
        description: 'Final tensor from Layer 1.',
        formulas: [{ label: 'Total Params', latex: '\\approx 7.1M', description: 'Per Layer' }],
        position: [0, 23.0, 0],
        dimensions: [3.5, DATA_HEIGHT, 3.5],
        color: C.DATA,
        dimLabel: '(B, S, 768)',
        group: 'layer_1'
      },
      
      // Summary
      {
        id: 'layers_n',
        title: 'Transformer Layers',
        type: 'output',
        category: 'summary',
        description: '11 more layers follow.',
        formulas: [{ label: 'Total', latex: '12 Layers', description: '' }],
        position: [0, 25.0, 0],
        dimensions: [3.5, 2, 3.5],
        color: C.SUMMARY,
        dimLabel: 'x 11 Layers'
      },
      
      {
        id: 'ctc_head',
        title: 'CTC Head',
        type: 'output',
        category: 'operation',
        description: 'Final classification (ASR).',
        formulas: [
            { label: 'Logits', latex: 'X W_{ctc} + b_{ctc}', description: 'Vocab projection' },
            { label: 'Loss', latex: 'L_{CTC}', description: 'Alignment Loss' }
        ],
        position: [0, 27, 0],
        dimensions: [3, 0.5, 3],
        color: C.OUTPUT,
        dimLabel: '768 x 32'
      }
    ]
  },

  [ModelType.PERSONA_PLEX]: {
    id: ModelType.PERSONA_PLEX,
    name: "PersonaPlex-7B (Moshi-based)",
    description: "Full-duplex speech-to-speech model using Mimi codec. Features Temporal Transformer (time) and Depth Transformer (RVQ stack).",
    specs: {
      hiddenSize: 4096,
      layers: 32,
      heads: 32,
      ffnSize: 11008,
      vocabSize: 32000,
      paramsPerLayer: "~202M",
      totalParams: "~7B",
      vramTraining: "~112 GB (Mixed Precision)",
    },
    stages: [
      // --- INPUTS ---
      {
        id: 'input_waveform',
        title: 'User Waveform',
        type: 'input',
        category: 'data',
        description: 'Raw audio input at 24kHz.',
        formulas: [{ label: 'Shape', latex: 'x_u \\in \\mathbb{R}^{B \\times T}', description: 'Time domain' }],
        position: [0, 0, 0],
        dimensions: [3, DATA_HEIGHT, 6],
        color: C.DATA,
        dimLabel: '(B, T)'
      },
      // --- MIMI ENCODER ---
      {
        id: 'mimi_encoder',
        title: 'Mimi Encoder',
        type: 'cnn',
        category: 'operation',
        description: 'Convolutional encoder + Transformer. 12.5Hz frame rate.',
        formulas: [
          { label: 'Enc', latex: 'h = \\text{Enc}_\\theta(x_u)', description: 'Latent extraction' },
          { label: 'Frame Rate', latex: 'r \\approx 12.5 \\text{Hz}', description: 'Low rate' }
        ],
        position: [0, 1.5, 0],
        dimensions: [3, OP_HEIGHT, 3],
        color: C.CNN,
        dimLabel: 'Conv + Trans'
      },
      {
        id: 'mimi_latent',
        title: 'Latent h',
        type: 'cnn',
        category: 'data',
        description: 'Continuous latent representation.',
        formulas: [{ label: 'Shape', latex: '(B, S, D_{lat})', description: 'S = r \\cdot t' }],
        position: [0, 2.5, 0],
        dimensions: [3, DATA_HEIGHT, 3],
        color: C.DATA,
        dimLabel: '(B, S, 512)'
      },
      // --- RVQ ---
      {
        id: 'rvq_quant',
        title: 'RVQ Quantization',
        type: 'cnn',
        category: 'operation',
        description: 'Residual Vector Quantization into 8 codebooks.',
        formulas: [
            { label: 'Proj', latex: 'u = h W_{qh} + b_{qh}', description: 'To D_q=256' },
            { label: 'Quant', latex: 'E_u \\in \\{0..2047\\}^{B \\times S \\times 8}', description: '8 levels' }
        ],
        position: [0, 3.5, 0],
        dimensions: [3, OP_HEIGHT, 3],
        color: C.CNN,
        dimLabel: 'Q=8 Codebooks'
      },
      {
        id: 'audio_tokens',
        title: 'Audio Tokens (E_u)',
        type: 'input',
        category: 'data',
        description: 'Discrete codec tokens.',
        formulas: [{ label: 'E_u', latex: '(B, S, 8)', description: 'Indices' }],
        position: [0, 4.5, 0],
        dimensions: [3, DATA_HEIGHT, 3],
        color: C.DATA,
        dimLabel: '(B, S, 8)'
      },

      // --- HYBRID PROMPT ASSEMBLY ---
      {
        id: 'embedding_lookup',
        title: 'Stream Embeddings',
        type: 'input',
        category: 'operation',
        description: 'Embed Audio (summed RVQ) and Text.',
        formulas: [
            { label: 'Audio', latex: 'X_{aud} = \\sum_{q=1}^8 W_{aud}^{(q)}[E^{(q)}]', description: 'Sum levels' },
            { label: 'Text', latex: 'X_{txt} = W_{txt}[T]', description: 'Text lookup' }
        ],
        position: [0, 5.5, 0],
        dimensions: [4, OP_HEIGHT, 4],
        color: C.INPUT,
        dimLabel: 'Embed Lookup'
      },
      {
        id: 'hybrid_prompt',
        title: 'Hybrid System Prompt',
        type: 'input',
        category: 'data',
        description: 'Concat: [Voice] || [Persona] || [Dialogue].',
        formulas: [{ label: 'X', latex: 'X_{voice} \\oplus X_{text} \\oplus X_{u}', description: 'Unified Stream' }],
        position: [0, 6.5, 0],
        dimensions: [4, DATA_HEIGHT, 4],
        color: C.DATA,
        dimLabel: '(B, S_total, 4096)'
      },

      // --- TEMPORAL TRANSFORMER (7B CORE) ---
      {
        id: 'temp_trans_block',
        title: 'Temporal Transformer',
        type: 'attention',
        category: 'operation',
        description: 'Time-axis processing (Moshi backbone).',
        formulas: [
            { label: 'Attn', latex: '\\text{Attention}(X)', description: 'Masked Time' },
            { label: 'MLP', latex: '\\text{SwiGLU}(X)', description: 'Feed Forward' }
        ],
        position: [0, 8.0, 0],
        dimensions: [4, 1.5, 4],
        color: C.ATTN,
        dimLabel: 'L=32, H=4096'
      },
      {
        id: 'temp_out',
        title: 'Temporal State (h_t)',
        type: 'attention',
        category: 'data',
        description: 'Contextualized state at time t.',
        formulas: [{ label: 'h_t', latex: '(B, S, 4096)', description: '' }],
        position: [0, 9.5, 0],
        dimensions: [4, DATA_HEIGHT, 4],
        color: C.DATA,
        dimLabel: '(B, S, 4096)'
      },

      // --- SPLIT OUTPUTS ---
      {
        id: 'heads_split',
        title: 'Duplex Branching',
        type: 'output',
        category: 'operation',
        description: 'Split to Text Head and Depth Transformer.',
        formulas: [],
        position: [0, 10.5, 0],
        dimensions: [6, 0.2, 0.2],
        color: C.OUTPUT,
        dimLabel: 'Branch'
      },

      // --- TEXT HEAD ---
      {
        id: 'text_head',
        title: 'Text Head',
        type: 'output',
        category: 'operation',
        description: 'Next token prediction for text.',
        formulas: [
            { label: 'Logits', latex: 'h_t W_{vocab}', description: 'Vocab ~32k' },
            { label: 'Loss', latex: 'L_{txt}', description: 'Cross Entropy' }
        ],
        position: [-2.5, 11.5, 0],
        dimensions: [2.5, OP_HEIGHT, 2.5],
        color: C.OUTPUT,
        dimLabel: 'Text Logits'
      },

      // --- AUDIO BRANCH (DEPTH TRANSFORMER) ---
      {
        id: 'depth_trans',
        title: 'Depth Transformer',
        type: 'cnn',
        category: 'operation',
        description: 'Autoregressive generation of 8 RVQ codes per frame.',
        formulas: [
            { label: 'Config', latex: 'L_d=6, H_d=1024', description: 'Small Trans.' },
            { label: 'AR', latex: 'p(E_q | h_t, E_{<q})', description: 'Inner loop' }
        ],
        position: [2.5, 11.5, 0],
        dimensions: [2.5, 1.5, 2.5],
        color: C.CNN, // Using CNN color for audio-related ops
        dimLabel: 'Depth Axis'
      },
      {
        id: 'audio_tokens_out',
        title: 'Pred. Audio Tokens (E_a)',
        type: 'cnn',
        category: 'data',
        description: 'Predicted agent audio tokens.',
        formulas: [{ label: 'E_a', latex: '(B, S, 8)', description: 'Indices' }],
        position: [2.5, 13.0, 0],
        dimensions: [2.5, DATA_HEIGHT, 2.5],
        color: C.DATA,
        dimLabel: '(B, S, 8)'
      },
      
      // --- MIMI DECODER ---
      {
        id: 'rvq_decode',
        title: 'RVQ Decode',
        type: 'cnn',
        category: 'operation',
        description: 'Lookup codebooks and sum vectors.',
        formulas: [
            { label: 'Sum', latex: '\\hat{u} = \\sum C^{(q)}[E_a^{(q)}]', description: 'Reconstruct' },
            { label: 'Proj', latex: '\\hat{h} = \\hat{u} W_{hq} + b', description: 'To Latent' }
        ],
        position: [2.5, 14.0, 0],
        dimensions: [2.5, OP_HEIGHT, 2.5],
        color: C.CNN,
        dimLabel: 'Lookup'
      },
      {
        id: 'mimi_decoder',
        title: 'Mimi Decoder',
        type: 'output',
        category: 'operation',
        description: 'Inverse ConvNet to Waveform.',
        formulas: [{ label: 'Dec', latex: '\\hat{x}_a = \\text{Dec}_\\phi(\\hat{h})', description: 'Waveform Gen' }],
        position: [2.5, 15.0, 0],
        dimensions: [2.5, OP_HEIGHT, 2.5],
        color: C.OUTPUT,
        dimLabel: 'Neural Vocoder'
      },
      {
        id: 'audio_out',
        title: 'Output Waveform',
        type: 'output',
        category: 'data',
        description: 'Synthesized speech.',
        formulas: [{ label: 'Audio', latex: '\\hat{x}_a \\in \\mathbb{R}^{B \\times T}', description: '24kHz' }],
        position: [2.5, 16.0, 0],
        dimensions: [2.5, DATA_HEIGHT, 2.5],
        color: C.DATA,
        dimLabel: '(B, T)'
      },
    ]
  }
};
