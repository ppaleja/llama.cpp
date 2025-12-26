# Copilot Rules: Adding a New Model

Purpose: give an automated agent clear, actionable steps and file-level guidance to add a new model architecture to this repo.

---

Prerequisites

- Know LLM basics, linear algebra, and C/C++ pointer/memory handling.
- Familiarize with GGML and Llama.cpp build/test workflows.

High-level workflow

1. Prepare tokenizer (if new).
2. Convert model tensors to GGUF using `convert_hf_to_gguf.py` (and updater script).
3. Register constants and tensor mappings.
4. Implement conversion class and preprocessing of tensors.
5. Add model loader hooks and graph builder (new `src/models` file + registration).
6. Implement graph builder using GGML ops and test against reference logits.
7. Debug, add prompt/chat templates if needed, and add tests.

Relevant files and responsibilities

- `convert_hf_to_gguf_update.py`: fetch/update tokenizers if the model uses a new tokenizer format; run before conversions when needed.
- `convert_hf_to_gguf.py` (gguf-py): primary conversion script. Implement or extend converter classes here. Use `constants.py` and `tensor_mapping.py` to align tensor names and types.
- `gguf-py/constants.py`: add model architecture constant, codename, and canonical list of tensors used by the architecture.
- `gguf-py/tensor_mapping.py`: map HF/safetensors parameter names to GGML names and types; add patterns for non-standard naming.
- `llama-arch.h` / `llama-arch.cpp`: add architecture identifier and any architecture-level helpers to mirror `constants.py`.
- `llama-model.cpp`: load hyperparameters in `load_hparams()` and load tensors in `load_tensors()`; update RoPE switching if needed.
- `src/models/` (new file): implement the graph builder for the architecture; add the file to `src/CMakeFiles.txt` and the class declaration to `src/models/models.h`.
- `src/models/models.h`: add declarations for the new model class.
- `examples/model-conversion`: scripts to compare logits between reference Transformer implementation and Llama.cpp; use these for numeric verification.
- `tools/` and `examples/llama-eval-callback`: helpers for tensor dumps and debugging.

Converter implementation: concrete methods to implement

- `set_gguf_parameters`: convert model hyperparameters into GGUF hparams.
- `prepare_tensors`: create or add special tensors needed by the GGML implementation (pack experts, precompute transforms, etc.).
- `modify_tensors`: rename, merge, omit, or transform tensors (e.g., merge expert tensors into single tensor, adjust weight/bias naming to `X.weight`/`X.bias`).

Tensor and shape rules

- GGML is little-endian row-major; reverse shapes from HF/Transformers when mapping (e.g., transpose semantics).
- GGML tensors are up to 4-D; pack higher dims manually.
- Matrix multiplication: `ggml_mul_mat(A,B)` corresponds to `transpose(B) @ A` from Transformers—watch transposes.
- Use `ggml_element_size`, `ggml_nelements`, `t->ne[]`, and `t->nb[]` for strides and sizes when creating views.

Graph builder guidelines

- Build a static inference graph using GGML ops; do not rely on runtime tensor values to change graph shape.
- Use existing GGML helpers for RoPE, attention, KV cache, LoRA, expert routing where applicable.
- Avoid generating loops in the graph builder; implement loops inside custom ops if necessary.

RoPE identification

- Inspect the reference `apply_rotary_pos_emb`:
  - If code slices and `repeat_interleave(2, dim=-1)` on cos/sin are present → normal RoPE.
  - If not present → NeoX RoPE.
- Mark appropriate RoPE type in `llama-model.cpp` switches.

Adding new GGML ops (if required)

- Add the queueing wrapper `ggml_...` (to schedule op in graph).
- Extend `ggml.h` / `ggml.c` enums and assertions.
- Implement CPU backend switch-case in `ggml-cpu.c` and op logic in `ops.cpp` / `unary-ops.cpp`.
- Add tests to `test-backend-ops`.

Testing & validation

- Use `examples/model-conversion` to compare logits with the reference model on short prompts.
- Run `llama-eval-callback` for detailed tensor dumps and single-token comparisons.
- Ensure `load_tensors()` loads every required tensor or explicitly ignores optional/unused tensors in `modify_tensors`.
- Run `ctest --test-dir build --output-on-failure` after building.

Debugging heuristics

- Mismatched outputs often come from incorrect transposes, shapes, or RoPE variants.
- Compare top-left and bottom-right corners of tensors across implementations to detect transposition errors quickly.
- If generation diverges but short-step logits match, inspect state management and RoPE hyperparameters.

Prompt templates & chat formats

- If the model uses non-standard prompt templates or tool markers, update `chat.cpp` and `llama-chat.cpp`:
  - Detect template fragments and map to parser rules
  - Add grammar and parser for tool-calling outputs if needed
  - Add tests to `test-chat.cpp` to validate parsing.

Performance & optimizations (post-correctness)

- After correctness, add backend optimizations (CUDA/Vulkan/Metal) and optimized kernels/shaders.
- Precompute tensor transforms (e.g., exp, scaling) during conversion in `modify_tensors` when only forward-pass is required.

Checklist for the conversion PR (minimal viable steps)

- [ ] Add tokenizer support via `convert_hf_to_gguf_update.py` if needed.
- [ ] Add constants and tensor mappings (`constants.py`, `tensor_mapping.py`).
- [ ] Implement converter class in `convert_hf_to_gguf.py` with `set_gguf_parameters`, `prepare_tensors`, `modify_tensors`.
- [ ] Add model loader entries in `llama-model.cpp`.
- [ ] Implement graph builder in `src/models/<new-model>.cpp` and declare in `src/models/models.h`.
- [ ] Add comparison scripts and run `examples/model-conversion` to validate.
- [ ] Add tests (tensor loading, chat parsing if needed, backend ops if new op added).

Notes for the agent

- Prefer forward-pass compatibility; avoid adding unnecessary backward-pass components.
- When uncertain about tensor naming, search `tensor_mapping.py` for similar patterns before adding new mappings.
- Keep changes minimal and consistent with the repo's coding style and tests.

---

Generated by Copilot-style rules: actionable, file-focused, and checklist-driven to guide an automated agent through adding a new model architecture to Llama.cpp.
