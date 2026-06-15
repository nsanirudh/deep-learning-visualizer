# Wav2Vec 2.0 (95M params)

A **CNN + Transformer** model for self-supervised speech representation learning, used for ASR (automatic speech recognition).

---

## Architecture Overview

| Component | Details |
|-----------|---------|
| CNN frontend | 7 × Conv1D layers, 512 channels, ~320× downsampling |
| Transformer | 12 blocks |
| Hidden size `H` | 768 |
| Attention heads `n_h` | 8 → head dim `d = 768/8 = 96` |
| FFN intermediate `F` | 3072 |
| Output vocab `V` | 32 CTC characters |

---

## Symbols

- `B` = batch size
- `T_raw` = raw waveform length (16 kHz samples)
- `S` = sequence length **after** CNN subsampling (latent frames)
- `H = 768` = transformer hidden size
- `n_h = 8`, `d = 96` = per-head dimension
- `F = 3072` = FFN intermediate size

---

## CNN Feature Extractor

Raw waveform at 16 kHz: **1 second → 16,000 samples → ~500 latent frames** (320× downsampling).

$$x \in \mathbb{R}^{B \times 1 \times T_{\mathrm{raw}}} \;\xrightarrow{\;\text{7 × Conv1D}\;}\; \mathbf{z} \in \mathbb{R}^{B \times 512 \times S}$$

Each convolutional layer reduces the time dimension via stride. After the stack:

$$\mathrm{rate} = 10\text{ ms/frame},\quad 10\text{ s audio} \;\Rightarrow\; S \approx 1000\text{ frames}$$

After transposing and projecting to the Transformer's hidden size:

$$\mathbf{z}^\top \in \mathbb{R}^{B \times S \times 512} \;\xrightarrow{W_{\mathrm{proj}} \in \mathbb{R}^{512 \times H}}\; X_0 \in \mathbb{R}^{B \times S \times H}$$

---

## Inside One Transformer Block

Input: $X \in \mathbb{R}^{B \times S \times H}$

### (A) Multi-Head Self-Attention

**Q, K, V projections** (with bias):

$$Q = XW_Q + b_Q,\quad K = XW_K + b_K,\quad V = XW_V + b_V$$

$$W_Q, W_K, W_V \in \mathbb{R}^{H \times H},\quad b_Q, b_K, b_V \in \mathbb{R}^{H}$$

**Reshape to $n_h$ heads:**

$$Q, K, V \;\rightarrow\; (B, n_h, S, d)$$

For Wav2Vec2-base: $H=768$, $n_h=8 \Rightarrow d=96$

$$Q : (B,\; 8,\; S,\; 96)$$

**Attention scores:**

$$A = \frac{QK^\top}{\sqrt{d}},\qquad A : (B, n_h, S, S)$$

Note: **no causal mask** — Wav2Vec uses **bidirectional** attention (full context).

**Weighted sum:**

$$P = \mathrm{softmax}(A),\qquad Z = PV \;\Rightarrow\; Z : (B, n_h, S, d)$$

**Merge + output projection:**

$$Z \;\rightarrow\; (B, S, H),\qquad Y = ZW_O + b_O,\quad W_O \in \mathbb{R}^{H \times H}$$

Each token/frame compares its query $q_t$ against all keys $k_1 \ldots k_S$ and mixes values — learning which audio frames are contextually related.

Attention params per layer (with bias):

$$P_{\mathrm{attn}} = 4(H^2 + H) = 4(768^2 + 768) = 4 \times 590{,}592 = 2{,}362{,}368$$

### (B) FFN Sublayer (standard, no SwiGLU)

$$\mathrm{FFN}(x) = W_2\,\sigma(W_1 x + b_1) + b_2$$

$$W_1 \in \mathbb{R}^{H \times F},\quad W_2 \in \mathbb{R}^{F \times H}$$

**Shape flow:**

$$X : (B, S, H) \;\xrightarrow{W_1}\; (B, S, F) \;\xrightarrow{\sigma}\; (B, S, F) \;\xrightarrow{W_2}\; (B, S, H)$$

FFN params per layer (with bias):

$$P_{\mathrm{ffn}} = 2HF + F + H = 2(768)(3072) + 3072 + 768 = 4{,}722{,}432$$

**Intuition:** Since $F \approx 4H$, the FFN is roughly $2H \cdot 4H = 8H^2$, which is **~2× larger than attention** ($\approx 4H^2$) per layer.

---

## Parameter Counting

$$P_{\mathrm{layer}} \approx P_{\mathrm{attn}} + P_{\mathrm{ffn}} \approx 2.4\mathrm{M} + 4.7\mathrm{M} \approx 7.1\mathrm{M}/\mathrm{layer}$$

With $L = 12$ blocks:

$$12 \times 7.1\mathrm{M} \approx 85\mathrm{M}$$

CNN frontend + projection + CTC head add the remaining ~10M, giving **~95M total**.

---

## VRAM (Training, AdamW Mixed Precision)

Same formula as any Transformer — 16 bytes/param for full fine-tuning:

$$\mathrm{VRAM}_{\mathrm{params}} \approx 16N \approx 16 \times 95\mathrm{M} \approx 1.5\text{ GB}$$

Activations scale with $B \cdot S \cdot H \cdot L$. For long audio sequences ($S \sim 1000$), activation memory can dominate despite the small model size.

---

## What the Model Learns

Wav2Vec 2.0 is trained to predict **quantized speech units** (discrete codebook entries) from masked regions of the audio:

$$\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(c_t, q_t)/\kappa)}{\sum_{q \in \tilde{Q}_t} \exp(\mathrm{sim}(c_t, q)/\kappa)}$$

where $c_t$ is the context vector (Transformer output), $q_t$ is the true quantized target, and $\tilde{Q}_t$ are distractors. This contrastive objective forces the model to capture phonetic structure without labels.
