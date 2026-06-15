# 7B Decoder-Only Transformer

A decoder-only Transformer with **~7 billion parameters**, used as the backbone of LLaMA-style large language models.

---

## Things That Matter in Memory

- **Weights** — trainable parameters (matrices, vectors)
- **Activations** — tensors produced in the forward pass that backward needs
- **Gradients** — $\frac{dL}{dW}$, same shape as weights
- **Optimizer states** — Adam stores $m$ and $v$ per param (often FP32)
- **Buffers / extras** — attention softmax buffers, GEMM workspace, fragmentation

---

## Symbols

| Symbol | Meaning | Value |
|--------|---------|-------|
| `B` | micro-batch size | — |
| `S` | sequence length (tokens) | — |
| `H` | hidden size | 4096 |
| `n_h` | attention heads | 32 |
| `d = H/n_h` | per-head dim | 128 |
| `I` | FFN intermediate size (SwiGLU) | 11008 |
| `L` | transformer blocks | 32 |
| `V` | vocabulary size | 32k |

Token IDs `(B, S)` → embedding → `(B, S, H)` → `L` transformer blocks → logits `(B, S, V)`

---

## Inside One Transformer Block

Given $X \in \mathbb{R}^{B \times S \times H}$:

### (A) Attention Sublayer

**Q, K, V projections** (no bias in LLaMA-style):

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$

$$W_Q, W_K, W_V \in \mathbb{R}^{H \times H}$$

**Reshape to heads** ($n_h$ heads, head dim $d = H / n_h$):

$$Q, K, V \;\rightarrow\; (B,\, n_h,\, S,\, d)$$

**Scaled dot-product attention** (causal mask applied):

$$A = \frac{QK^\top}{\sqrt{d}} \;\Rightarrow\; A : (B, n_h, S, S)$$

$$P = \mathrm{softmax}(A),\qquad Z = PV \;\Rightarrow\; Z : (B, n_h, S, d)$$

**Merge heads + output projection:**

$$Z \;\rightarrow\; (B, S, H),\qquad Y = ZW_O,\quad W_O \in \mathbb{R}^{H \times H}$$

### (B) SwiGLU FFN Sublayer

Common in modern LLMs — three projection matrices instead of two:

$$\mathrm{SwiGLU}(x) = \bigl(xW_{\mathrm{up}}\bigr) \odot \sigma\!\bigl(xW_{\mathrm{gate}}\bigr)$$

$$\text{output} = \mathrm{SwiGLU}(x)\cdot W_{\mathrm{down}}$$

$$W_{\mathrm{up}},\; W_{\mathrm{gate}}: H \rightarrow I,\qquad W_{\mathrm{down}}: I \rightarrow H$$

### (C) Residual + Pre-RMSNorm

$$X \;\leftarrow\; X + \mathrm{Attn}\!\bigl(\mathrm{RMSNorm}(X)\bigr)$$

$$X \;\leftarrow\; X + \mathrm{FFN}\!\bigl(\mathrm{RMSNorm}(X)\bigr)$$

---

## Parameter Breakdown Per Layer

$$P_{\mathrm{attn}} \approx 4H^2 = 4 \times 4096^2 \approx 67\mathrm{M}$$

$$P_{\mathrm{ffn}} \approx 3HI = 3 \times 4096 \times 11008 \approx 135\mathrm{M}$$

$$\boxed{P_{\mathrm{layer}} \approx 4H^2 + 3HI \approx 202\mathrm{M}}$$

With $L = 32$ blocks:

$$202\mathrm{M} \times 32 \approx 6.47\mathrm{B} \text{ params in transformer blocks alone}$$

**Embedding / output head** (often tied):

$$W_{\mathrm{embed}} \in \mathbb{R}^{32000 \times 4096} \approx 131\mathrm{M}$$

This gives the **~7B neighborhood**.

---

## VRAM in Full Fine-Tuning (AdamW, Mixed Precision)

Per-parameter memory breakdown:

| Component | Bytes/param |
|-----------|-------------|
| Weights (BF16/FP16) | 2 |
| Gradients (BF16/FP16) | 2 |
| Adam $m$ moment (FP32) | 4 |
| Adam $v$ moment (FP32) | 4 |
| FP32 master weights | 4 |
| **Total** | **16** |

$$\mathrm{VRAM}_{\mathrm{params}} \approx 16N \approx 16 \times 7\mathrm{B} \approx 112\text{ GB}$$

### Per-Layer Breakdown

- Weights (BF16): $202\mathrm{M} \times 2 \approx 404\text{ MB}$
- Gradients (BF16): $202\mathrm{M} \times 2 \approx 404\text{ MB}$
- Adam moments (FP32): $202\mathrm{M} \times 8 \approx 1.6\text{ GB}$
- FP32 master weights: $202\mathrm{M} \times 4 \approx 808\text{ MB}$

**Total per layer ≈ 3.23 GB × 32 ≈ 103 GB** (just params/grads/optimizer)

### Activations

The residual stream per layer has shape $(B, S, H)$. At BF16, one layer costs:

$$B \times S \times H \times 2 \text{ bytes}$$

**Example** $B=1$, $S=2048$, $H=4096$:

$$1 \times 2048 \times 4096 \times 2 = 16\text{ MB/layer} \times 32 \approx 512\text{ MB}$$

But training saves more than just the residual — pre/post-norm, MLP intermediates, attention tensors. The attention score matrix is especially expensive:

$$A : (B, n_h, S, S) \quad \text{materialises } B \times 32 \times S^2 \text{ floats}$$

Use **FlashAttention** to avoid materialising $A$ and reduce activation memory significantly.

> **Activations** $\approx \text{(a few ×)}\; L \cdot B \cdot S \cdot H \cdot \text{bytes}$ — the exact multiplier depends on implementation and whether you use gradient checkpointing.
