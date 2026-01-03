# MotifForCausalLM Implementation Progress

Issue: [#18055](https://github.com/ggml-org/llama.cpp/issues/18055)

Model: MotifForCausalLM (Motif-2-12.7B-Reasoning)
Reference: [Hugging Face modeling_motif.py](https://huggingface.co/Motif-Technologies/Motif-2-12.7B-Reasoning/blob/main/modeling_motif.py)
vLLM Reference: [Motif branch](https://github.com/MotifTechnologies/vllm/tree/motif)

**Status: ✅ COMPLETE - Model is working!**

**Key Architectural Features:**
- GroupedDifferentialAttention (custom attention scheme)
- PolyNorm (custom normalization)

---

## Phase 1: Preparation & Analysis ✅ COMPLETE

> **Documentation:** See [phase-1.md](./phase-1.md) for full research findings.

### Summary

- [x] Tokenizer: BPE-based, Llama-3/Tiktoken compatible (already supported)
- [x] Architecture: `MotifForCausalLM` with 40 layers
- [x] Unique Features Identified:
  - **GroupedDifferentialAttention (GDA):** Requires custom graph with learnable `lambda` parameters
  - **PolyNorm:** Learnable polynomial activation with `act_fn.weight` and `act_fn.bias`
  - **SubLayerNorm:** Internal normalization in attention (`subln.weight`)
- [x] RoPE Type: Standard RoPE with high theta (1,000,000)
- [x] Tensor packing: Lambda scalars need correct handling

---

## Phase 2: Tensor Conversion ✅ COMPLETE

### Constants & Mappings

- [x] Added `MOTIF` architecture constant to `gguf-py/gguf/constants.py`
- [x] Added tensor name mappings to `gguf-py/gguf/tensor_mapping.py`:
  - `lambda_q1/k1/q2/k2` → `ATTN_LAMBDA_Q1/K1/Q2/K2`
  - `act_fn.weight/bias` → `FFN_POLYNORM_W/B`
  - `subln.weight` → `ATTN_SUB_NORM`
- [x] Added Motif-specific hyperparameter keys:
  - `NUM_NOISE_HEADS`, `GROUPED_RATIO`, `K_RATIO`, `LAMBDA_INIT`

### Converter Implementation

- [x] Created `MotifModel` class in `convert_hf_to_gguf.py`
- [x] Implemented `set_gguf_parameters()` with `super()` call for RoPE handling
- [x] Fixed: Removed spurious `ATTN_LAMBDA_INIT` tensor (correctly stored as hyperparameter only)

---

## Phase 3: Model Loading ✅ COMPLETE

### Architecture Registration

- [x] Add `LLM_ARCH_MOTIF` to `llama-arch.h`
- [x] Add `"motif"` string mapping in `llama-arch.cpp`

### Tensor Type Registration

- [x] Add `LLM_TENSOR_ATTN_LAMBDA_Q1/K1/Q2/K2` to tensor enum
- [x] Add `LLM_TENSOR_ATTN_SUB_NORM` to tensor enum
- [x] Add `LLM_TENSOR_FFN_POLYNORM_W/B` to tensor enum

### Layer Struct Extension

- [x] Add Motif-specific tensors to `llama_layer` struct in `llama-model.h`:
  - `lambda_q1/k1/q2/k2` (differential attention parameters)
  - `attn_sub_norm` (sublayer normalization)
  - `ffn_polynorm_w/b` (PolyNorm activation parameters)

### Tensor Loading

- [x] Add `LLM_ARCH_MOTIF` case in `load_tensors()` with all Motif-specific tensors

### Verification

- [x] Load GGUF file without "skipping unknown tensor" warnings
- [x] Verify hyperparameters are parsed correctly

---

## Phase 4: Graph Builder Implementation ✅ COMPLETE

> **Analysis**: See [phase-4-analysis.md](./phase-4-analysis.md) for detailed breakdown.

### New Model File & Registration

- [x] Create `src/models/motif.cpp` (graph builder)
- [x] Add class declaration to `src/models/models.h`
- [x] Add file to `src/CMakeLists.txt`

### Graph Construction

- [x] Implement core graph builder class `llm_build_motif`
- [x] Token embedding
- [x] RoPE normalization (standard `ggml_rope_ext` with `LLAMA_ROPE_TYPE_NORM`)
- [x] Layer stack:
  - [x] GroupedDifferentialAttention
    - [x] Lambda computation (`ggml_mul`, `ggml_sum`, `ggml_exp`)
    - [x] Q/K/V head splitting with grouped pattern (matching HuggingFace `_reshape_heads`)
    - [x] K/V head expansion with `repeat_interleave_heads` helper
    - [x] Fused Q/K/V concatenation for two parallel attention calls
    - [x] Differential calculation: `attn_o - lambda_full * attn_n`
    - [x] SubLayerNorm (applied before flattening)
  - [x] PolyNorm (built from primitives: `ggml_pow`, `ggml_sqr`, `ggml_rms_norm`)
  - [x] FFN layers (standard MLP structure with PolyNorm activation)
- [x] Output projection & logits
- [x] KV cache handling (standard approach with valid context slicing)

---

## Phase 5: Debugging & Bug Fixes ✅ COMPLETE

### Initial Issues

**Problem:** Model produced garbage output (random noise, then repetitive text, then semi-coherent garbage).

### Bug Fixes Applied

#### Fix 1: Reshape Order Bug
- **Issue:** Used `ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_tokens, n_head)` - wrong order
- **Fix:** Corrected to `ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens)`
- **Impact:** Critical - incorrect tensor layout broke all subsequent operations

#### Fix 2: SubLayerNorm Timing
- **Issue:** Applied `attn_sub_norm` AFTER flattening (wrong shape: `[8192, N]` instead of `[256, 32, N]`)
- **Fix:** Apply before flattening, on correct shape
- **Impact:** High - shape mismatch prevented proper normalization

#### Fix 3: Head Repeat Semantics (K/V)
- **Issue:** Used `ggml_repeat` (tiling: `[A,B,C,D] → [A,B,C,D,A,B,C,D]`)
- **HuggingFace:** Uses `repeat_interleave` (interleaving: `[A,B,C,D] → [A,A,A,A,B,B,B,B,C,C,C,C,D,D,D,D]`)
- **Fix:** Implemented `repeat_interleave_heads()` helper function
- **Impact:** Critical - head ordering was scrambled

#### Fix 4: Lambda Init Calculation
- **Issue:** Used constant `lambda_init = 0.0` for all layers
- **HuggingFace:** `lambda_init = 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))`
- **Fix:** Implemented per-layer calculation
- **Impact:** High - differential attention balance was wrong

#### Fix 5: Head Repeat Semantics (Attention Output)
- **Issue:** `attn_n` used `ggml_repeat` (tiling) instead of `repeat_interleave`
- **Fix:** Used `repeat_interleave_heads()` for `attn_n` expansion
- **Impact:** Critical - head alignment in differential subtraction was wrong

#### Fix 6: Non-Contiguous Tensor
- **Issue:** `attn_n_group` is a non-contiguous view, caused assertion failure in `ggml_reshape_4d`
- **Fix:** Added `ggml_cont()` before `repeat_interleave_heads()`
- **Impact:** Critical - crashed during batch inference

#### Fix 7: FFN Numerical Precision
- **Issue:** Missing float32 upcast before PolyNorm (HuggingFace uses `.float()`)
- **Fix:** Added `ggml_cast` to F32 before PolyNorm, then cast back
- **Impact:** Medium - improved numerical stability for x³ terms

### Debugging Process

1. **KV Cache Investigation:** Verified KV cache slicing was correct (not the root cause)
2. **Attention Mask Analysis:** Confirmed causal mask construction was standard
3. **Differential Bypass Test:** Isolated that base attention had bugs independent of differential logic
4. **Systematic Comparison:** Created detailed comparison document against HuggingFace implementation
5. **Head Split Verification:** Confirmed grouped pattern implementation matched `einops.rearrange`
6. **RoPE Validation:** Verified standard RoPE type and parameters

### Verification Results

**Before fixes:**
```
Output: "and get the imagination of official good life, did. I Do Get Donece row 5..."
```

**After fixes:**
```
> goodbye
Okay, the user just said "goodbye". I should respond politely 
and wish them a good day. I should keep it friendly and positive.

> hi there, what's up?
Okay, the user said "hi there, what's up?" So I need to respond 
appropriately. The user is greeting me and asking how I'm doing.
```

✅ **Model now generates coherent, context-aware English responses!**

---

## Phase 6: Testing & Validation ✅ COMPLETE

### Build & Compilation

- [x] CMake build successful
- [x] No compilation errors
- [x] Model loads without warnings

### Generation Testing

- [x] Basic inference works: `./build/bin/llama-cli -m model.gguf -p "goodbye" -n 32`
- [x] Output is coherent and context-aware
- [x] Different prompts produce different, appropriate responses
- [x] Model shows internal reasoning (meta-commentary style typical of instruction-tuned models)

---

## Technical Implementation Details

### Helper Functions Implemented

#### `repeat_interleave_heads()`
Implements PyTorch's `repeat_interleave` semantics for head expansion:
```cpp
static ggml_tensor * repeat_interleave_heads(ggml_context * ctx, ggml_tensor * x, int n_rep) {
    // Reshape [d, heads, seq] → [d, heads, 1, seq]
    // Repeat [d, heads, 1, seq] → [d, heads, n_rep, seq]
    // Permute [d, heads, n_rep, seq] → [d, n_rep, heads, seq]
    // Reshape [d, n_rep, heads, seq] → [d, heads*n_rep, seq]
}
```

#### `build_polynorm()`
Implements PolyNorm activation from primitives:
```
output = weight[0] * RMSNorm(x³) + weight[1] * RMSNorm(x²) + weight[2] * RMSNorm(x) + bias
```

### Key Architecture Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_heads` | 40 | Total Q heads |
| `num_key_value_heads` | 16 | Total K/V heads (GQA) |
| `num_noise_heads` | 8 | Noise heads for differential |
| `grouped_ratio` | 4 | Ratio of signal to noise heads |
| `k_ratio` | 1 | K head grouping ratio |
| `head_dim` | 128 | Dimension per head |
| `rope_theta` | 1,000,000 | High theta for long context |

### Differential Attention Flow

1. **Split Q:** `Q (40 heads) → Q1 (32 heads) + Q2 (8 heads)` [grouped pattern]
2. **Split K/V:** `K (16 heads) → K1 (8) + K2 (8)`, `V (16) → V1 (8) + V2 (8)` [grouped pattern]
3. **Expand K/V:** K1/V1 expanded 4x with `repeat_interleave` → 32 heads each
4. **Fuse:** 
   - `q_f = concat(Q1, Q2)` → 40 heads
   - `k_f = concat(K1_expanded, K2)` → 40 heads
   - `v1_f = concat(V1_expanded, V1)` → 40 heads
   - `v2_f = concat(V2_expanded, V2)` → 40 heads
5. **Attention:** `attn_1 = attention(q_f, k_f, v1_f)`, `attn_2 = attention(q_f, k_f, v2_f)`
6. **Merge & Split:** 
   - Concat `[attn_1, attn_2]` along head_dim → `[256, 40, N]`
   - `attn_o = merged[:, :32, :]` → 32 heads
   - `attn_n_group = merged[:, 32:, :]` → 8 heads
   - `attn_n = repeat_interleave(attn_n_group, 4)` → 32 heads
7. **Differential:** `output = attn_o - lambda_full * attn_n`
8. **SubLayerNorm:** Apply RMSNorm, scale by `(1 - lambda_init)`
9. **Output Projection:** Flatten to 8192, apply `wo` → 4096

---

## Files Modified

### Core Implementation
- `src/models/motif.cpp` - Graph builder with differential attention
- `src/models/models.h` - Class declaration
- `src/CMakeLists.txt` - Build system

### Architecture & Loading
- `src/llama-arch.h` - Architecture enum
- `src/llama-arch.cpp` - Tensor mappings
- `src/llama-model.h` - Layer struct extension
- `src/llama-model.cpp` - Tensor loading logic

### Conversion
- `gguf-py/gguf/constants.py` - MOTIF architecture constant
- `gguf-py/gguf/tensor_mapping.py` - Tensor name mappings
- `convert-hf-to-gguf.py` - MotifModel converter class

---

## Lessons Learned

1. **Repeat vs Repeat Interleave:** Critical difference between tiling and interleaving for head expansion
2. **Tensor Contiguity:** Always check if views are contiguous before operations requiring it
3. **Shape Ordering:** GGML dimension ordering differs from PyTorch - verify carefully
4. **Per-Layer Parameters:** Hyperparameters can vary by layer (e.g., `lambda_init`)
5. **Numerical Precision:** Float32 upcast matters for stability in polynomial operations
6. **Systematic Comparison:** Creating detailed comparison documents helps track subtle differences

---

## Resources

- **HuggingFace Model:** [Motif-2-12.7B-Reasoning](https://huggingface.co/Motif-Technologies/Motif-2-12.7B-Reasoning)
- **vLLM Implementation:** [MotifTechnologies/vllm](https://github.com/MotifTechnologies/vllm/tree/motif)
- **llama.cpp Guide:** [Adding a New Model Architecture](https://github.com/ggml-org/llama.cpp/discussions/16770)
- **Issue:** [#18055](https://github.com/ggml-org/llama.cpp/issues/18055)

---

*Last updated: 2026-01-03*
*Status: Implementation complete and working ✅*
