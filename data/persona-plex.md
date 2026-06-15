# PersonaPlex (7.2B) — Speech-to-Speech

A **speech-to-speech** model that combines a large LLM decoder backbone with an audio codec (Mimi encoder + RVQ quantizer + Mimi decoder) to convert voice input directly to synthesised voice output.

---

## Architecture Overview

```
Voice input (waveform)
  │
  ▼
Mimi Encoder  (CNN vocoder)
  │  continuous latents  (B, S, 512)
  ▼
RVQ Quantizer  (8 codebooks × 2048 entries)
  │  discrete audio tokens  (B, S, 8)
  ▼
Embedding lookup  (8 × 2048 × 4096)
  │
  ├──────────────────────────────┐
  │                              │
  ▼                              ▼
LLM Backbone (32-layer decoder) ← text prompt / style context
  │  hidden states  (B, S, H)
  ├── Text head  → next-token logits  (B, S, 32000)
  └── Depth transformer  → audio code predictions  (B, 8, 1024)
         │
         ▼
      RVQ Decode  →  Mimi Decoder  →  Synthesised waveform
```

---

## Symbols

| Symbol | Meaning | Value |
|--------|---------|-------|
| `B` | batch size | — |
| `S` | sequence length | — |
| `H` | LLM hidden size | 4096 |
| `n_h` | attention heads | 32 |
| `d = H/n_h` | per-head dim | 128 |
| `I` | FFN intermediate (SwiGLU) | 11008 |
| `L` | transformer blocks | 32 |
| `K` | RVQ codebooks | 8 |
| `E` | codebook entries | 2048 |

---

## Mimi Audio Codec

### Encoder (CNN)

Raw waveform at 24 kHz is compressed to discrete tokens via strided convolutions:

$$\text{waveform} \in \mathbb{R}^{B \times T_{\mathrm{raw}}} \;\xrightarrow{\text{CNN}}\; \mathbf{z} \in \mathbb{R}^{B \times S \times 512}$$

### Residual Vector Quantization (RVQ)

The continuous latents are quantized through $K=8$ codebooks in sequence. Each codebook $k$ has $E=2048$ entries of dimension 256:

$$\mathbf{z} \approx \sum_{k=1}^{K} e_k^{(i_k)}, \quad i_k \in \{1 \ldots E\}$$

At each stage, the current residual is mapped to its nearest codebook entry, which is subtracted before the next stage:

$$r_0 = \mathbf{z}$$
$$i_k = \arg\min_{j} \|r_{k-1} - e_k^{(j)}\|_2$$
$$r_k = r_{k-1} - e_k^{(i_k)}$$

This produces **8 discrete indices per frame**: audio tokens $(B, S, 8)$.

### Embedding Lookup

Each of the 8 codebook indices is projected to the LLM's hidden size:

$$W_{\mathrm{embed}} \in \mathbb{R}^{K \times E \times H} = 8 \times 2048 \times 4096$$

The $K$ embeddings are summed to produce the per-frame input to the LLM.

---

## LLM Backbone (same as 7B Decoder)

The 7.2B parameter decoder uses the same architecture as a standard LLaMA-style model. Each of the 32 blocks applies:

### Attention

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

$$A = \frac{QK^\top}{\sqrt{d}},\quad P = \mathrm{softmax}(A),\quad Z = PV$$

$$Y = ZW_O, \qquad \text{shapes: } (B, n_h, S, d) \text{ throughout}$$

### SwiGLU FFN

$$\mathrm{SwiGLU}(x) = \bigl(xW_{\mathrm{up}}\bigr) \odot \sigma\!\bigl(xW_{\mathrm{gate}}\bigr)$$

$$\text{output} = \mathrm{SwiGLU}(x) \cdot W_{\mathrm{down}}, \qquad W_{\mathrm{up}}, W_{\mathrm{gate}}: H \to I,\quad W_{\mathrm{down}}: I \to H$$

### Pre-RMSNorm + Residual

$$X \leftarrow X + \mathrm{Attn}(\mathrm{RMSNorm}(X))$$
$$X \leftarrow X + \mathrm{FFN}(\mathrm{RMSNorm}(X))$$

---

## Dual Output Heads

The LLM produces two outputs in parallel:

**Text head** (next-token LM prediction):

$$W_{\mathrm{text}} \in \mathbb{R}^{H \times V_{\mathrm{text}}},\quad \ell_{\mathrm{text}} : (B, S, 32000)$$

**Depth transformer** (audio code prediction, one per RVQ level):

$$\text{depth features} : (B, K, 1024) \;\xrightarrow{\mathrm{heads}}\; \text{code logits per codebook}$$

---

## Parameter Breakdown

$$P_{\mathrm{attn}} \approx 4H^2 \approx 67\mathrm{M/layer}$$

$$P_{\mathrm{ffn}} \approx 3HI \approx 135\mathrm{M/layer}$$

$$P_{\mathrm{layer}} \approx 202\mathrm{M}, \quad P_{\mathrm{blocks}} \approx 202\mathrm{M} \times 32 \approx 6.47\mathrm{B}$$

Codec + embeddings + dual heads account for the remaining ~730M, giving **~7.2B total**.

---

## VRAM (Training, AdamW Mixed Precision)

Using the standard 16 bytes/param formula:

| Component | Bytes/param |
|-----------|-------------|
| Weights (BF16) | 2 |
| Gradients (BF16) | 2 |
| Adam $m$ + $v$ (FP32) | 8 |
| FP32 master weights | 4 |
| **Total** | **16** |

$$\mathrm{VRAM}_{\mathrm{params}} \approx 16 \times 7.2\mathrm{B} \approx 115\text{ GB}$$

The RVQ quantizer introduces an additional $(S \times U)$ lattice structure similar to RNN-T that must be respected during backpropagation.
