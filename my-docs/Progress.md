# MotifForCausalLM Implementation Progress

Issue: [#18055](https://github.com/ggml-org/llama.cpp/issues/18055)

Model: MotifForCausalLM (Motif-2-12.7B-Reasoning)  
Reference: [Hugging Face modeling_motif.py](https://huggingface.co/Motif-Technologies/Motif-2-12.7B-Reasoning/blob/main/modeling_motif.py)  
vLLM Reference: [Motif branch](https://github.com/MotifTechnologies/vllm/tree/motif)

**Key Architectural Features:**

- GroupedDifferentialAttention (custom attention scheme)
- PolyNorm (custom normalization)

---

## Phase 1: Preparation & Analysis

### Tokenizer & Model Inspection

- [ ] Identify tokenizer type used by MotifForCausalLM
- [ ] Check if tokenizer is already supported in llama.cpp
  - If new: add to `convert_hf_to_gguf_update.py`
  - If existing: skip this step
- [ ] Download `model.safetensors.index.json` and `config.json` from reference model
- [ ] Create checklist of all model tensors in `model.safetensors.index.json`
- [ ] Review `modeling_motif.py` to understand layer structure and parameters
- [ ] Review vLLM reference for additional implementation details

### Architecture Decisions

- [ ] Identify if GroupedDifferentialAttention can use existing GGML attention primitives
  - If not, plan custom GGML operation(s)
- [ ] Identify if PolyNorm can be implemented as a variant of existing layer norm
  - If not, plan custom GGML operation(s)
- [ ] Determine if any expert/specialized tensor packing is needed
- [ ] Identify RoPE type (normal vs NeoX) from `apply_rotary_pos_emb` in reference code

---

## Phase 2: Tensor Conversion

### Constants & Mappings

- [ ] Add `MOTIF` architecture constant to `gguf-py/constants.py`
  - Include codename and full tensor list
- [ ] Add tensor name mappings to `gguf-py/tensor_mapping.py`
  - Map HF parameter names to GGML convention
  - Handle non-standard naming if present (e.g., `_weight`, `_bias` vs `.weight`, `.bias`)
- [ ] Update `llama-arch.h` and `llama-arch.cpp` with architecture identifier

### Converter Implementation

- [ ] Create converter class in `convert_hf_to_gguf.py`
  - Inherit from appropriate base class (likely `TextModel` or similar)
  - Implement `set_gguf_parameters()` for hyperparameters
  - Implement `prepare_tensors()` for any special tensor creation/packing
  - Implement `modify_tensors()` for renaming, merging, omitting tensors
- [ ] Test conversion with reference model
  - Run converter and verify all tensors load without errors
  - Check tensor dimensions match expected shapes

---

## Phase 3: Model Loading

### Hyperparameter & Tensor Loading

- [ ] Add hyperparameter loading in `llama-model.cpp` `load_hparams()`
  - Extract Motif-specific parameters from GGUF
- [ ] Add tensor loading in `llama-model.cpp` `load_tensors()`
  - Handle all converted tensors
  - Explicitly ignore any unused tensors (e.g., if not implementing training)
- [ ] Verify model size enum is updated (if needed)
- [ ] Test successful model loading with a reference model file

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
