# MotifForCausalLM Implementation Progress

Issue: [#18055](https://github.com/ggml-org/llama.cpp/issues/18055)

Model: MotifForCausalLM (Motif-2-12.7B-Reasoning)
Reference: [Hugging Face modeling_motif.py](https://huggingface.co/Motif-Technologies/Motif-2-12.7B-Reasoning/blob/main/modeling_motif.py)
vLLM Reference: [Motif branch](https://github.com/MotifTechnologies/vllm/tree/motif)

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

> **Verification:** See [GGUF_Conversion.ipynb](../GGUF_Conversion.ipynb) for test results.

### Constants & Mappings

- [x] Added `MOTIF` architecture constant to `gguf-py/gguf/constants.py` (line 458)
- [x] Added tensor name mappings to `gguf-py/gguf/tensor_mapping.py`:
  - `lambda_q1/k1/q2/k2` → `ATTN_LAMBDA_Q1/K1/Q2/K2`
  - `act_fn.weight/bias` → `FFN_POLYNORM_W/B`
  - `subln.weight` → `ATTN_SUB_NORM`
- [x] Added Motif-specific hyperparameter keys:
  - `NUM_NOISE_HEADS`, `GROUPED_RATIO`, `K_RATIO`, `LAMBDA_INIT`

### Converter Implementation

- [x] Created `MotifModel` class in `convert_hf_to_gguf.py` (line 10560)
- [x] Implemented `set_gguf_parameters()` with `super()` call for RoPE handling
- [x] Fixed: Removed spurious `ATTN_LAMBDA_INIT` tensor (correctly stored as hyperparameter only)

### Verification Results (from notebook)

| Check | Result |
|-------|--------|
| Total tensors | 643 |
| `attn_lambda_q1/k1/q2/k2` | 40 each ✅ |
| `attn_sub_norm` | 40 ✅ |
| `ffn_polynorm_w/b` | 40 each ✅ |
| `grouped_ratio` | 4.0 ✅ |
| `num_noise_heads` | 8 ✅ |
| `lambda_init` | 1.0 (hyperparameter) ✅ |

---

## Phase 3: Model Loading 🔄 NEXT

### Compatibility Analysis

| Component | Motif Implementation | llama.cpp Status |
|-----------|---------------------|------------------|
| RoPE | Standard with high theta (1M) | ✅ Reusable |
| RMSNorm | Standard | ✅ Reusable |
| MLP structure | gate→act→up→down | ✅ Reusable |
| **PolyNorm** | `w[0]*norm(x³) + w[1]*norm(x²) + w[2]*norm(x) + b` | ❌ Custom op needed |
| **Differential Attention** | `attn1 - λ*attn2` with learnable λ | ❌ Custom graph needed |
| **SubLayerNorm** | RMSNorm inside attention | ❌ Custom placement |
| **Lambda tensors** | 4 per-layer vectors (128 dims) | ❌ New tensor type |

### Objective (Minimal)

Register Motif architecture and enable tensor loading. Defer custom attention/activation to Phase 4.

### Tasks

#### Architecture Registration (Minimal)

- [ ] Add `LLM_ARCH_MOTIF` to `llama-arch.h`
- [ ] Add `"motif"` string mapping in `llama-arch.cpp`

#### Tensor Type Registration

- [ ] Add `LLM_TENSOR_ATTN_LAMBDA_Q1/K1/Q2/K2` to tensor enum
- [ ] Add `LLM_TENSOR_ATTN_SUB_NORM` to tensor enum
- [ ] Add `LLM_TENSOR_FFN_POLYNORM_W/B` to tensor enum

#### Layer Struct Extension

- [ ] Add to `llama_layer` struct in `llama-model.h`:
  ```cpp
  // Motif differential attention
  ggml_tensor * lambda_q1 = nullptr;
  ggml_tensor * lambda_k1 = nullptr;
  ggml_tensor * lambda_q2 = nullptr;
  ggml_tensor * lambda_k2 = nullptr;
  ggml_tensor * attn_sub_norm = nullptr;
  
  // Motif PolyNorm
  ggml_tensor * ffn_polynorm_w = nullptr;
  ggml_tensor * ffn_polynorm_b = nullptr;
  ```

#### Tensor Loading

- [ ] Add `LLM_ARCH_MOTIF` case in `load_tensors()` (copy from LLAMA, add new tensors)

#### Verification

- [ ] Load GGUF file without "skipping unknown tensor" warnings
- [ ] Verify hyperparameters are parsed

### Files to Modify

| File | Changes |
|------|---------|
| `src/llama-arch.h` | Add enum value |
| `src/llama-arch.cpp` | Add tensor mappings |
| `src/llama-model.h` | Extend layer struct |
| `src/llama-model.cpp` | Add tensor loading case |

---


## Phase 4: Graph Builder Implementation

### New Model File & Registration

- [ ] Create `src/models/motif.cpp` (graph builder)
- [ ] Add class declaration to `src/models/models.h`
- [ ] Add file to `src/CMakeLists.txt`

### Graph Construction

- [ ] Implement core graph builder class with `build()` method
  - Token embedding
  - Positional encoding (RoPE)
  - Layer stack:
    - [ ] GroupedDifferentialAttention (or use fallback attention + plan custom op)
    - [ ] PolyNorm (or use fallback layer norm + plan custom op)
    - [ ] FFN layers
  - Output projection & logits
- [ ] Mark output tensor with `ggml_build_forward_expand()`
- [ ] Handle KV cache for inference (if applicable)

### Custom Operations (if needed)

- [ ] If GroupedDifferentialAttention requires custom op:
  - [ ] Design GGML operation signature
  - [ ] Add operation to `ggml.h` / `ggml.c`
  - [ ] Implement CPU backend in `ggml-cpu.c` / `ops.cpp`
  - [ ] Add tests to `test-backend-ops.cpp`
- [ ] If PolyNorm requires custom op:
  - [ ] Follow same steps as attention custom op

### RoPE Configuration

- [ ] Verify RoPE type from reference `apply_rotary_pos_emb`
  - If interleave + repeat_interleave(2) present: normal RoPE
  - If not: NeoX RoPE
- [ ] Add RoPE switch case in `llama-model.cpp` if needed

---

## Phase 5: Testing & Validation

### Reference Comparison

- [ ] Run `examples/model-conversion/run-org-model` with reference Transformers model
- [ ] Generate logits for short prompt (e.g., 1-5 tokens)
- [ ] Compare Llama.cpp logits with reference logits
  - Use tolerance (e.g., 1e-4 for float32)
  - Debug mismatches by comparing intermediate tensor dumps

### Tensor Debugging (if needed)

- [ ] Use `llama-eval-callback` for single-token processing
- [ ] Compare tensor dumps at each layer
- [ ] Check for transposition errors (compare top-left & bottom-right corners)
- [ ] Verify shape correctness in tensor views

### Build & Unit Tests

- [ ] Build with CMake: `cmake -B build && cmake --build build --config Release`
- [ ] Run model loading test
- [ ] Run all tests: `ctest --test-dir build --output-on-failure`
- [ ] Verify no test failures (or only expected network-related failures)

### Generation Testing

- [ ] Test basic inference with `llama-cli`:

  ```bash
  ./build/bin/llama-cli -m model.gguf -p "Hello" -n 10
  ```

- [ ] Test long-context inference (if applicable)
- [ ] Verify coherent output generation

---

## Phase 6: Chat Templates & Prompts (if needed)

### Check for Non-Standard Templates

- [ ] Review Motif model card for custom prompt template
- [ ] Check if special thinking markers or tool-call syntax is used
- [ ] If non-standard detected:
  - [ ] Add chat format detection in `chat.cpp`
  - [ ] Add grammar rules in `llama-chat.cpp` if needed
  - [ ] Add parser for tool/special outputs
  - [ ] Add tests to `test-chat.cpp`

---

## Phase 7: Performance & Optimization (Post-Correctness)

- [ ] Profile inference on representative workloads
- [ ] Add backend-specific optimizations:
  - [ ] CUDA kernels (if applicable)
  - [ ] Metal shaders (if applicable)
  - [ ] Vulkan shaders (if applicable)
- [ ] Optimize tensor transforms in `modify_tensors()` (precompute if possible)

---

## Phase 8: Documentation & PR

### Code Quality

- [ ] Run `git clang-format` on all C++ files
- [ ] Verify Python code with pre-commit hooks (flake8, pyright)

### Testing Checklist Before PR

- [ ] All CMake builds pass (Release & Debug on supported platforms)
- [ ] `ctest` passes (or documents expected failures)
- [ ] Model conversion works end-to-end
- [ ] Reference model logits match within tolerance
- [ ] Generation produces coherent output

### PR Submission

- [ ] Create feature branch
- [ ] Write clear commit messages
- [ ] Open PR with reference to #18055
- [ ] Link model conversion tests and results

---

## Notes & Resources

- **Copilot Instructions**: Refer to `.github/instructions/copilot-adding-new-model.instructions.md`
- **Guide**: [Adding a New Model Architecture (discussion #16770)](https://github.com/ggml-org/llama.cpp/discussions/16770)
- **Reference Models**:
  - HF: [Motif-2-12.7B-Reasoning](https://huggingface.co/Motif-Technologies/Motif-2-12.7B-Reasoning)
  - vLLM Motif branch: [MotifTechnologies/vllm](https://github.com/MotifTechnologies/vllm/tree/motif)
- **Troubleshooting**:
  - Divergence in logits: check transposes, RoPE type, attention variant
  - Tensor shape mismatches: verify GGML little-endian reversal
  - Custom op issues: add tests to `test-backend-ops` early

---

*Last updated: 2025-12-26*
