# Phase 4: Graph Building Analysis for Motif Model

## Overview

This document analyzes the Motif architecture to determine which tensor operations can be reused from GGML and which require custom implementation.

---

## ✅ **Operations That Can Be Reused**

### 1. **Standard Tensor Operations**

All basic tensor operations are available in GGML:

- **Matrix multiplication**: `ggml_mul_mat()` for Q/K/V/O projections
- **Element-wise operations**: `ggml_add()`, `ggml_mul()`, `ggml_sub()`, `ggml_scale()`
- **Reshape/View**: `ggml_reshape_3d()`, `ggml_view_4d()`, `ggml_cont()`
- **Transpose/Permute**: `ggml_permute()`, `ggml_transpose()`
- **Unary ops**: `ggml_exp()`, `ggml_sqrt()`, `ggml_sum()` (for lambda computation)
- **Broadcasting**: Built into `ggml_mul()`, `ggml_add()`, etc.

### 2. **RoPE (Rotary Position Embedding)**

**Status**: ⚠️ **NEEDS VERIFICATION** - Type unclear

From `modeling_motify.py`:

**MotifRotaryEmbeddingWithCache** (lines 105-158):
- Computes frequencies: `freqs = torch.outer(t, inv_freq)` (line 144)
- Concatenates: `emb = torch.cat((freqs, freqs), dim=-1)` (line 146)
- Creates cos/sin cache from concatenated frequencies

**apply_rotary_pos_emb** (lines 267-293):
- Uses `rotate_half()` function (standard pattern)
- Does NOT use `repeat_interleave(2)` on cos/sin before applying (lines 291-293)
- Commented code (lines 280-286) shows what would be normal RoPE with `repeat_interleave(2)`

**According to llama.cpp docs** (`adding-a-new-model.md`):
- If `repeat_interleave(2)` is present → **normal** RoPE
- If `repeat_interleave(2)` is absent → **NeoX** RoPE

**Analysis**:
- The actual implementation does NOT have `repeat_interleave(2)` → suggests **NeoX** RoPE
- However, `torch.cat((freqs, freqs))` at the frequency level may achieve similar effect

**vLLM PR #27396 Findings**:
- `MotifForCausalLM` inherits from `LlamaForCausalLM` in vLLM
- Uses vLLM's existing RoPE primitives (doesn't implement custom RoPE)
- Since Llama models in vLLM typically use **normal** RoPE, this suggests Motif likely uses **normal** RoPE
- However, the Transformers implementation pattern suggests NeoX

**Recommendation**: 
- Start with **normal** RoPE (since vLLM uses it and Motif inherits from Llama)
- Test both types during implementation to determine correct one
- The `torch.cat((freqs, freqs))` pattern may be equivalent to interleaving at a different stage

**GGML equivalent**: `ggml_rope_ext()` - can be used, but need to determine `rope_type` parameter:

```cpp
// Try both and verify which produces correct logits:
// Option 1: Normal RoPE
Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                     n_rot, LLAMA_ROPE_TYPE_NORMAL, n_ctx_orig, freq_base, freq_scale,
                     ext_factor, attn_factor, beta_fast, beta_slow);

// Option 2: NeoX RoPE  
Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                     n_rot, LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, freq_base, freq_scale,
                     ext_factor, attn_factor, beta_fast, beta_slow);
```

### 3. **RMSNorm**

**Status**: ✅ Fully reusable

From `modeling_motify.py` lines 85-99, `MotifRMSNorm` is standard RMSNorm:
- Computes: `x * rsqrt(mean(x²) + eps)`
- Same as standard RMSNorm used in LLaMA

**GGML equivalent**: `ggml_rms_norm()` - can be used directly

```cpp
cur = ggml_rms_norm(ctx0, cur, eps);
cur = ggml_mul(ctx0, cur, weight);  // element-wise multiply by weight
```

### 4. **Standard Attention Components**

**Status**: ✅ Partially reusable (Q/K/V projections, softmax, but not the full attention)

- **Q/K/V projections**: `build_lora_mm()` for linear projections
- **Softmax**: `ggml_soft_max_ext()` for attention weights
- **Matrix multiplication**: `ggml_mul_mat()` for Q@K^T and attn@V

**Note**: The full attention mechanism is custom (differential attention), but individual components can be reused.

### 5. **MLP/FFN Structure**

**Status**: ✅ Structure reusable, activation needs custom

The MLP structure (`gate → act → up → down`) is standard:
- Gate projection: `ggml_mul_mat()`
- Up projection: `ggml_mul_mat()`
- Down projection: `ggml_mul_mat()`

**However**: The activation function (PolyNorm) is custom and needs implementation.

### 6. **KV Cache Handling**

**Status**: ✅ Fully reusable

Standard KV cache operations:
- `build_attn_inp_kv()` for KV cache input
- Cache storage/retrieval via `llama_kv_cache` methods

---

## ❌ **Operations That Need Custom Implementation**

### 1. **PolyNorm Activation** 🔴 **REQUIRES CUSTOM OP**

**Location**: `modeling_motify.py` lines 54-77

**Formula**:
```python
def forward(self, x):
    return (
        self.weight[0] * self._norm(x**3) +
        self.weight[1] * self._norm(x**2) +
        self.weight[2] * self._norm(x) +
        self.bias
    )

def _norm(self, x):
    return x / sqrt(mean(x.pow(2), dim=-1, keepdim=True) + eps)
```

**Why custom op needed**:
- Complex operation combining polynomial terms with normalization
- Each term (`x³`, `x²`, `x`) needs normalization before weighting
- Normalization is per-row (along last dimension)
- More efficient as a fused operation than building from primitives

**Implementation options**:

**Option A: Build from primitives (easier, less efficient)**
```cpp
// Compute x^3, x^2, x
ggml_tensor * x3 = ggml_pow(ctx0, x, 3.0f);
ggml_tensor * x2 = ggml_sqr(ctx0, x);
ggml_tensor * x1 = x;

// Normalize each (per-row normalization)
ggml_tensor * n3 = normalize_rowwise(ctx0, x3, eps);
ggml_tensor * n2 = normalize_rowwise(ctx0, x2, eps);
ggml_tensor * n1 = normalize_rowwise(ctx0, x1, eps);

// Weight and sum
ggml_tensor * w3 = ggml_scale(ctx0, n3, weight[0]);
ggml_tensor * w2 = ggml_scale(ctx0, n2, weight[1]);
ggml_tensor * w1 = ggml_scale(ctx0, n1, weight[2]);

ggml_tensor * result = ggml_add(ctx0, w3, w2);
result = ggml_add(ctx0, result, w1);
result = ggml_add(ctx0, result, bias);
```

**Option B: Custom GGML operation (more efficient)**
- Add `GGML_OP_POLYNORM` to `ggml.h`
- Implement CPU backend in `ggml-cpu/ops.cpp`
- Add tests to `test-backend-ops.cpp`

**Recommendation**: Start with Option A for correctness, optimize to Option B later.

### 2. **GroupedDifferentialAttention** 🔴 **REQUIRES CUSTOM GRAPH**

**Location**: `modeling_motify.py` lines 317-521

**Key operations**:

1. **Lambda computation** (lines 497-499):
   ```python
   lambda_1 = exp(sum(lambda_q1 * lambda_k1, dim=-1))
   lambda_2 = exp(sum(lambda_q2 * lambda_k2, dim=-1))
   lambda_full = lambda_1 - lambda_2 + lambda_init
   ```

2. **Differential attention** (lines 500-501):
   ```python
   attn_weights = attn_weights.view(bsz, num_heads, 2, q_len, -1)
   attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
   ```

3. **SubLayerNorm** (line 504):
   ```python
   attn_output = subln(attn_output)  # RMSNorm on (2 * head_dim)
   attn_output = attn_output * (1 - lambda_init)
   ```

4. **Head reshaping** (lines 406-420):
   - Complex head splitting/grouping operations using `einops.rearrange`
   - Splits Q/K/V into grouped and noise heads

**Why custom graph needed**:
- Lambda computation involves element-wise product and sum reduction
- Attention weights need to be reshaped to `[bsz, heads, 2, q_len, kv_len]` for differential computation
- Two separate attention computations (attn1, attn2) that are then combined
- SubLayerNorm applied to concatenated output `[2 * head_dim]`

**Implementation approach**:

**Step 1: Lambda computation** (can use existing ops)
```cpp
// lambda_q1/k1 are [head_dim] tensors
ggml_tensor * lambda_q1k1 = ggml_mul(ctx0, lambda_q1, lambda_k1);
ggml_tensor * lambda_sum1 = ggml_sum(ctx0, lambda_q1k1);  // sum along last dim
ggml_tensor * lambda_1 = ggml_exp(ctx0, lambda_sum1);

// Same for lambda_2
ggml_tensor * lambda_q2k2 = ggml_mul(ctx0, lambda_q2, lambda_k2);
ggml_tensor * lambda_sum2 = ggml_sum(ctx0, lambda_q2k2);
ggml_tensor * lambda_2 = ggml_exp(ctx0, lambda_sum2);

// Combine
ggml_tensor * lambda_full = ggml_add(ctx0, 
    ggml_sub(ctx0, lambda_1, lambda_2),
    ggml_new_f32(ctx0, lambda_init));
```

**Step 2: Head reshaping** (needs custom view operations)
- Use `ggml_view_4d()` or `ggml_view_3d()` to split Q/K/V
- Pattern: `einops.rearrange(tensor, "... (num_groups group_size) D -> ... num_groups group_size D")`
- This is complex but can be done with views and reshapes

**Step 3: Differential attention** (custom graph)
```cpp
// Compute two attention maps (attn1, attn2)
ggml_tensor * attn1 = build_standard_attention(q1, k1, v1, ...);
ggml_tensor * attn2 = build_standard_attention(q2, k2, v2, ...);

// Reshape to [bsz, heads, 2, q_len, kv_len]
ggml_tensor * attn_concat = ggml_concat(ctx0, attn1, attn2, /*dim=*/2);

// Extract and compute differential
ggml_tensor * attn_o = ggml_view_4d(...);  // attn_concat[:, :, 0, :, :]
ggml_tensor * attn_n = ggml_view_4d(...);  // attn_concat[:, :, 1, :, :]

// Differential: attn_o - lambda_full * attn_n
ggml_tensor * lambda_broadcast = ggml_scale(ctx0, attn_n, lambda_full);
ggml_tensor * attn_diff = ggml_sub(ctx0, attn_o, lambda_broadcast);
```

**Step 4: SubLayerNorm** (can reuse RMSNorm)
```cpp
// attn_output shape: [bsz, heads, q_len, 2 * head_dim]
ggml_tensor * attn_normed = ggml_rms_norm(ctx0, attn_output, eps);
attn_normed = ggml_mul(ctx0, attn_normed, subln_weight);
attn_normed = ggml_scale(ctx0, attn_normed, 1.0f - lambda_init);
```

**Recommendation**: Implement as custom graph builder function `build_differential_attention()` in `motif.cpp`. No new GGML ops needed, but complex tensor manipulation.

### 3. **SubLayerNorm Placement**

**Status**: ⚠️ Custom placement, but uses standard RMSNorm

The SubLayerNorm is applied **inside** the attention block (after attention output, before output projection). This is just a matter of graph structure, not a new operation.

---

## 📋 **Implementation Strategy**

### Phase 4A: Graph Structure (No Custom Ops)

1. **Implement basic graph structure**:
   - Token embedding
   - Layer loop with standard components
   - Use fallback implementations for custom parts

2. **PolyNorm**: Build from primitives initially
   - Use `ggml_pow()`, `ggml_sqr()`, `ggml_rms_norm()`, `ggml_add()`, `ggml_mul()`
   - Verify correctness first

3. **Differential Attention**: Build custom graph
   - Implement `build_differential_attention()` function
   - Use existing attention building blocks
   - Handle head reshaping with views/reshapes

### Phase 4B: Optimization (Optional Custom Ops)

1. **PolyNorm custom op**: If performance is an issue
   - Add `GGML_OP_POLYNORM`
   - Fuse normalization and polynomial computation

2. **Differential Attention fusion**: If needed
   - Could fuse lambda computation and attention subtraction
   - Likely not necessary - graph structure is sufficient

---

## 🔍 **Key Implementation Details**

### Lambda Tensor Handling

From `modeling_motify.py` lines 391-393:
- `lambda_q1/k1/q2/k2` are `[head_dim]` tensors (128 dims)
- They are learnable parameters, not computed
- Stored as `ggml_tensor *` in layer struct

### Head Dimensions

From `modeling_motify.py` lines 366-368:
- `num_noise_heads = 8`
- `grouped_ratio = 4.0` (meaning 4:1 ratio)
- `q_heads = (grouped_ratio + 1) * num_noise_heads = 5 * 8 = 40` total heads
- But `num_heads = 32` (from config), so there's a mismatch - need to verify

### Value Tensor Structure

From `modeling_motify.py` line 385:
- `v_proj` outputs `2 * k_noise_heads * head_dim` (two value streams)
- This is split into `v1` and `v2` for differential attention

---

## ✅ **Summary Table**

| Component | Reusable? | Notes |
|-----------|-----------|-------|
| RoPE | ✅ Yes | Standard `ggml_rope_ext()` |
| RMSNorm | ✅ Yes | Standard `ggml_rms_norm()` |
| Q/K/V projections | ✅ Yes | Standard `ggml_mul_mat()` |
| Softmax | ✅ Yes | Standard `ggml_soft_max_ext()` |
| Matrix ops | ✅ Yes | All basic ops available |
| KV cache | ✅ Yes | Standard cache handling |
| **PolyNorm** | ❌ No | Build from primitives or custom op |
| **Differential Attention** | ❌ No | Custom graph builder function |
| **Head reshaping** | ⚠️ Partial | Complex but doable with views |
| **SubLayerNorm** | ✅ Yes | Standard RMSNorm, custom placement |

---

## 🎯 **Next Steps**

1. **Start with Phase 4A**: Build graph structure using existing ops
2. **Implement PolyNorm from primitives**: Verify correctness
3. **Implement differential attention graph**: Use existing attention building blocks
4. **Test and validate**: Compare logits with reference implementation
5. **Optimize if needed**: Add custom ops only if performance requires it

---

*Last updated: 2025-12-26*

