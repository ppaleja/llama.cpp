User: I want to work on this issue (adding support for MotifForCausalLM) (https://github.com/ggml-org/llama.cpp/issues/18055). Here is how support was added to vLLM https://github.com/vllm-project/vllm/pull/27396. Here is recent PR adding support to llama.cpp for LlaDA2.0, maybe it will help https://github.com/ggml-org/llama.cpp/pull/17454. Make a plan for implementing this feature

GitHub Copilot: Enable GPT-5.1-Codex-Max (Preview) for all clients
[object Object]

User: @agent Enable: "Enable GPT-5.1-Codex-Max (Preview) for all clients"

GitHub Copilot: I'll research the issue and reference implementations to create a comprehensive plan for adding MotifForCausalLM support.

Research MotifForCausalLM implementation

Fetching item [#18055](https://github.com/ggml-org/llama.cpp/issues/18055) from GitHub
Completed with input: {
  "issueNumber": 18055,
  "repo": {
    "owner": "ggml-org",
    "name": "llama.cpp"
  }
}

Using "Fetch Web Page"

Fetched https://github.com/vllm-project/vllm/pull/27396

Fetching item [#17454](https://github.com/ggml-org/llama.cpp/issues/17454) from GitHub
Completed with input: {
  "issueNumber": 17454,
  "repo": {
    "owner": "ggml-org",
    "name": "llama.cpp"
  }
}

Searched codebase for "LLM_ARCH model architecture registration enum", 10 results

Using "Fetch Web Page"

Fetched https://huggingface.co/Motif-Technologies/Motif-2-12.7B-Reasoning/blob/main/modeling_motif.py

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/src/llama-model.cpp#1-1), lines 1 to 100

Searched for regex `load_hparams|LLM_ARCH_.*:` (`**/src/llama-model.cpp`), 20 results

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/src/llama-arch.h#1-1), lines 1 to 150

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/convert_hf_to_gguf.py#1-1), lines 1 to 100

Searched for regex `class.*Model\(|@ModelBase.register` (`**/convert_hf_to_gguf.py`), 20 results

Searched codebase for "Differential attention implementation grouped noise heads lambda parameters", 30 results

Using "Fetch Web Page"

Fetched https://huggingface.co/Motif-Technologies/Motif-2-12.7B-Reasoning/blob/main/config.json

User: continue planning

GitHub Copilot: Using "Fetch Web Page"

Fetched 3 resources

Let me fetch the Motif model architecture file directly and examine existing model implementations in llama.cpp:

Using "Fetch Web Page"

Fetched https://huggingface.co/Motif-Technologies/Motif-2-12.7B-Reasoning/blob/main/modeling_motif.py

Searched codebase for "LLM_ARCH model architecture enum registration new model type", no results

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/src/llama.cpp#1-1), lines 1 to 200

Let me now search for existing model architecture implementations and examine the converter to understand the pattern better:

Searched for text `LLM_ARCH_` (`**/src/llama.cpp`), 1 result

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/convert_hf_to_gguf.py#1-1), lines 1 to 150

Searched for regex `class.*Model.*Writer` (`**/convert_hf_to_gguf.py`), no results

Searched codebase for "model architecture enum definition LLM_ARCH llama qwen phi", no results

Searched for regex `class \w+Model` (`**/convert_hf_to_gguf.py`), 30 results

Searched for files matching `**/*llada*`, 2 matches

Perfect! Now let me examine the LLaDA implementation to understand the pattern:

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/src/models/llada.cpp)

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/convert_hf_to_gguf.py#200-200), lines 200 to 3497

Searched for text `LLADA` (`**/gguf-py/gguf/constants.py`), 8 results

Perfect! Now let me create a comprehensive implementation plan based on the information gathered:

## Implementation Plan for MotifForCausalLM Support in llama.cpp

Based on my analysis of the Motif architecture, vLLM implementation, and llama.cpp patterns (particularly LLaDA), here's the comprehensive plan:

### **1. Architecture Overview**

**Motif Model** uses **Grouped Differential Attention** which is a novel attention mechanism that:
- Computes attention as the difference between two separate softmax attention maps
- Uses learnable lambda parameters (`lambda_q1`, `lambda_k1`, `lambda_q2`, `lambda_k2`, `lambda_init`)
- Groups queries into origin heads and noise heads
- Applies a `subln` (Sub Layer Normalization) on attention outputs
- Uses PolyNorm activation function for MLP (custom polynomial-based normalization)

### **2. Key Differences from Standard LLaMA**
1. **Attention Mechanism**: Differential attention instead of standard scaled dot-product
2. **Custom Activation**: PolyNorm (polynomial normalization) instead of SiLU/GELU
3. **Additional Normalization**: Sub-layer RMSNorm within attention blocks
4. **Lambda Parameters**: Learnable scalars for differential attention
5. **Head Grouping**: Special handling of `num_noise_heads` and `grouped_ratio`

### **3. Implementation Steps**

#### **Phase 1: Define Model Architecture**

**File**: constants.py
- Add `MOTIF = auto()` to `MODEL_ARCH` enum
- Add `MODEL_ARCH.MOTIF: "motif"` to `MODEL_ARCH_NAMES`
- Define tensor keys for Motif-specific parameters:
  ```python
  MODEL_TENSOR.ATTN_LAMBDA_Q1
  MODEL_TENSOR.ATTN_LAMBDA_K1
  MODEL_TENSOR.ATTN_LAMBDA_Q2
  MODEL_TENSOR.ATTN_LAMBDA_K2
  MODEL_TENSOR.ATTN_LAMBDA_INIT
  MODEL_TENSOR.ATTN_SUBLN  # Sub-layer norm
  MODEL_TENSOR.FFN_POLYNORM  # PolyNorm weights/bias
  ```
- Add tensor name mappings in `MODEL_TENSORS` dict
- Add metadata keys:
  ```python
  KEY.ATTENTION.NUM_NOISE_HEADS
  KEY.ATTENTION.GROUPED_RATIO
  KEY.ATTENTION.K_RATIO
  KEY.ATTENTION.LAMBDA_INIT
  ```

#### **Phase 2: Converter Implementation**

**File**: convert_hf_to_gguf.py

Add MotifModel class:
```python
@ModelBase.register("MotifForCausalLM")
class MotifModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MOTIF

    def set_vocab(self):
        # Motif uses GPT-2 style tokenizer
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        # Standard parameters
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hparams["num_key_value_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])

        # Motif-specific parameters
        self.gguf_writer.add_uint32(f"{self.arch}.attention.num_noise_heads",
                                    hparams["num_noise_heads"])
        self.gguf_writer.add_uint32(f"{self.arch}.attention.k_ratio",
                                    hparams.get("k_ratio", 1))

        # RoPE parameters
        if hparams.get("head_dim"):
            self.gguf_writer.add_key_length(hparams["head_dim"])
            self.gguf_writer.add_value_length(hparams["head_dim"])

        # Rope theta
        if (rope_theta := hparams.get("rope_theta")):
            self.gguf_writer.add_rope_freq_base(rope_theta)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Handle lambda parameters (per-layer scalars)
        if "lambda_q1" in name or "lambda_k1" in name or \
           "lambda_q2" in name or "lambda_k2" in name:
            # These are learned scalars per layer
            return [(self.map_tensor_name(name), data_torch)]

        # Handle subln (sub-layer normalization)
        if "subln" in name:
            return [(self.map_tensor_name(name), data_torch)]

        # Handle PolyNorm activation weights/bias
        if "act_fn.weight" in name or "act_fn.bias" in name:
            return [(self.map_tensor_name(name), data_torch)]

        # Standard tensor handling
        return [(self.map_tensor_name(name), data_torch)]
```

#### **Phase 3: C++ Model Implementation**

**File**: `src/models/motif.cpp` (new file)

Create the build function:
```cpp
llm_build_motif::llm_build_motif(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd = hparams.n_embd;
    const int64_t n_head = hparams.n_head;
    const int64_t n_head_kv = hparams.n_head_kv;
    const int64_t n_noise_heads = hparams.n_noise_heads;

    // Grouped differential attention architecture
    const int64_t grouped_ratio = (n_head - n_noise_heads) / n_noise_heads;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // Pre-attention norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

        // Differential attention computation
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);

            // Apply RoPE
            Qcur = ggml_rope_ext(...);
            Kcur = ggml_rope_ext(...);

            // Implement differential attention logic here
            // This involves:
            // 1. Computing two separate attention maps
            // 2. Applying lambda weighting
            // 3. Subtracting the second from the first
            // 4. Applying sub-layer norm

            cur = build_differential_attention(
                KQ_mask, model.layers[il].wo,
                Qcur, Kcur, Vcur,
                model.layers[il].lambda_q1, model.layers[il].lambda_k1,
                model.layers[il].lambda_q2, model.layers[il].lambda_k2,
                model.layers[il].lambda_init, model.layers[il].subln,
                n_noise_heads, grouped_ratio, il);
        }

        // Residual connection
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);

        // Feed-forward with PolyNorm activation
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);

        // FFN with custom PolyNorm activation
        cur = build_ffn_polynorm(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                model.layers[il].polynorm_weight,
                model.layers[il].polynorm_bias, il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        inpL = cur;
    }

    // Output norm and LM head
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cur = build_lora_mm(model.output, cur);

    ggml_build_forward_expand(gf, cur);
}
```

#### **Phase 4: Model Loading**

**File**: llama-model.cpp

Add to model loader:
```cpp
case LLM_ARCH_MOTIF:
    {
        // Load standard parameters
        model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

        // Load lambda parameters per layer
        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.lambda_q1 = create_tensor(tn(LLM_TENSOR_ATTN_LAMBDA_Q1, "weight", i), {n_embd_head});
            layer.lambda_k1 = create_tensor(tn(LLM_TENSOR_ATTN_LAMBDA_K1, "weight", i), {n_embd_head});
            layer.lambda_q2 = create_tensor(tn(LLM_TENSOR_ATTN_LAMBDA_Q2, "weight", i), {n_embd_head});
            layer.lambda_k2 = create_tensor(tn(LLM_TENSOR_ATTN_LAMBDA_K2, "weight", i), {n_embd_head});

            layer.subln = create_tensor(tn(LLM_TENSOR_ATTN_SUBLN, "weight", i), {2 * n_embd_head});

            // PolyNorm parameters
            layer.polynorm_weight = create_tensor(tn(LLM_TENSOR_FFN_POLYNORM, "weight", i), {3});
            layer.polynorm_bias = create_tensor(tn(LLM_TENSOR_FFN_POLYNORM, "bias", i), {1});

            // Standard Q, K, V, O projections
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), ...);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), ...);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), ...);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), ...);

            // FFN
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), ...);
            layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), ...);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), ...);

            // Norms
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), ...);
            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), ...);
        }
    } break;
```

#### **Phase 5: Implement Differential Attention Operation**

**File**: ggml.c or custom operations in llama-impl.cpp

Need to implement:
1. **Grouped differential attention kernel**
2. **PolyNorm activation function**
3. **Sub-layer normalization**

### **4. Testing Strategy**

1. **Unit Tests**: Test individual components
   - PolyNorm activation
   - Differential attention computation
   - Lambda parameter application

2. **Integration Tests**:
   - Convert a small Motif model
   - Run perplexity tests
   - Compare outputs with HuggingFace implementation

3. **Validation**:
   - Test with official Motif-2-12.7B model
   - Verify numerical accuracy
   - Benchmark performance

### **5. Documentation**

Update:
- build.md - Add Motif support notes
- Model architecture docs
- Conversion guide
- Example usage

### **6. Files to Create/Modify**

**New Files:**
- `src/models/motif.cpp`
- `tests/test-motif.cpp` (optional)

**Modified Files:**
- constants.py
- convert_hf_to_gguf.py
- llama-model.cpp
- llama.cpp
- llama.h (if new APIs needed)
- CMakeLists.txt (add motif.cpp)

### **7. Potential Challenges**

1. **Differential Attention Complexity**: The grouped differential attention is more complex than standard attention
2. **PolyNorm Implementation**: Custom activation function needs efficient implementation
3. **Memory Layout**: Proper handling of grouped heads and noise heads
4. **Quantization**: May need special handling for lambda parameters

### **8. Timeline Estimate**

- **Phase 1-2** (Constants & Converter): 1-2 days
- **Phase 3-4** (C++ Implementation): 3-5 days
- **Phase 5** (Custom Operations): 2-3 days
- **Testing & Validation**: 2-3 days
- **Documentation**: 1 day

**Total: ~2 weeks** for a complete implementation

This plan provides a solid foundation for implementing Motif support. The key is following the existing patterns (like LLaDA) while properly handling the unique differential attention mechanism.

User: We need to find what all is already there though right? Can you search thru the repo to find what is already that and what we need to reimplement and propose adjustments to the plan

GitHub Copilot: Perfect — proceeding to scan the repository for any existing Motif-related code, PolyNorm, differential-attention primitives, and converter/gguf support so I can adjust the plan. I'll return a concise inventory of findings and recommended changes.

Search and summarise repository contents relevant to Motif support

Searched for regex `motif|poly_norm|polynorm|lambda_q|lambda_k|grouped_ratio|num_noise_heads|differential attention|subln|Grouped Differential|apply_rotary_pos_emb|MotifForCausalLM|MOTIF`, 3 results

Searched for regex `flash|sdpa|eager attention|rotary|rope|apply_rotary|apply_rotary_pos_emb|rmsnorm|RMSNorm|T5LayerNorm|PolyNorm|poly_norm|polynorm`, 200 results

Searched for regex `grouped|grouped_head|grouped_heads|grouped_ratio|num_noise_heads|differential attention|differential`, 76 results

Searched for regex `PolyNorm|poly_norm|polynorm|lambda_q|lambda_k|subln|SubLN|"subln"`, no results

Searched for regex `RMSNorm|RMS Norm|rmsnorm|RootMeanSquare|T5LayerNorm|T5 LayerNorm|t5layernorm`, 27 results

Searched for regex `sdpa|scaled_dot_product|sdpa attention|sdpa`, no results

Searched for regex `attn|attention`, 200 results

Searched for regex `apertus|apertus_mod|MotifForCausalLM|Motif`, 74 results

Searched for regex `LLaDA|LLaDA-MoE|Dream|DreamForCausalLM|MoE|Mixture of Experts|afmoe|bailingmoe`, 200 results

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/examples/model-conversion/scripts/causal/run-org-model.py#1-1), lines 1 to 80

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/src/llama-graph.cpp#1348-1348), lines 1348 to 1386

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh#760-760), lines 760 to 804

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/tests/test-backend-ops.cpp#5798-5798), lines 5798 to 5822

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/gguf-py/gguf/constants.py#430-430), lines 430 to 460

Searched for regex `lambda_q|lambda_k|lambda`, 91 results

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/src/llama-model.h#192-192), lines 192 to 214

Searched for regex `class llm_build_dream|dream` (`**/src/**`), 17 results

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/convert_hf_to_gguf.py#9754-9754), lines 9754 to 9784

Searched for text `add_xielu|add_ssm_dt_b_c_rms|add_` (`**/gguf-py/**`), 200 results

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/convert_llama_ggml_to_gguf.py#380-380), lines 380 to 410

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/convert_hf_to_gguf.py#6156-6156), lines 6156 to 6176

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/src/llama-model.cpp#6338-6338), lines 6338 to 6392

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/src/llama-hparams.h#116-116), lines 116 to 156

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/gguf-py/gguf/tensor_mapping.py#1378-1378), lines 1378 to 1394

Perfect! I scanned the repo and found useful existing infrastructure and also clear gaps. Below is a concise actionable summary, adjusted plan, and priorities so we can proceed efficiently.

---

## 🔎 Findings (high-confidence) ✅
- Reusable infrastructure:
  - Flash attention + grouped-query attention (GQA) kernels (CUDA/Vulkan) — useful for Motif grouped heads. ✅
  - RoPE helpers and GGUF rope metadata (rope_freqs, scaling) — can reuse for Motif RoPE variants. ✅
  - RMSNorm/T5-like norm primitives and per-layer norm slots in `llama_model` — helpful for SubLN/PolyNorm integration. ✅
  - Converter hook pattern (`@ModelBase.register`, `modify_tensors`, tensor mapping) already exists and is straightforward to extend. ✅
  - GGUF writer supports adding metadata / keys — easy to add Motif-specific fields. ✅
  - Unit test scaffolding for flash/GQA exists — good place to add motif tests. ✅
- Missing / needs implementation:
  - No occurrences of "Motif", "MotifForCausalLM", "poly_norm"/"polynorm", "lambda_q"/"lambda_k", "num_noise_heads", or "grouped_ratio" — these must be added. ⚠️
  - No PolyNorm implementation (C++/ggml) — needs a primitive or composed op. ⚠️
  - No grouped differential attention (Motif) kernel or graph wiring — needs CPU path first, GPU/flash later. ⚠️

---

## ✍️ Adjusted Plan & Priorities (short, 3–6 steps) 🔧

### Plan: Add MotifForCausalLM support (updated)
TL;DR: Add converter + GGUF metadata (quick unblock), add PolyNorm primitive & tests (low-risk core), implement Motif CPU attention graph (functional testing), then extend to GPU/flash kernels and add integration tests.

### Steps (priority order)
1. Add `MotifForCausalLM` converter class in convert_hf_to_gguf.py to map Motif-specific tensors (`lambda_q*`, `lambda_k*`, `lambda_init`, `subln`, PolyNorm params) and write gguf metadata. (Files: convert_hf_to_gguf.py, constants.py) ✅
2. Add gguf constants & tensor name mappings for Motif (enum, MODEL_ARCH.MOTIF, tensor names for lambdas/subln/polynorm). (Files: constants.py) ✅
3. Add unit tests verifying conversion & gguf keys (Python tests). (Files: `gguf-py/tests/test_motif_conversion.py`) ✅
4. Implement PolyNorm (reference implementation + CPU ggml op or composed ops) with unit tests. (Files: ggml or relevant ggml backend + `tests/test-poly-norm.cpp`) 🔧
5. Add motif graph builder `llm_build_motif` and minimal CPU attention implementation that computes grouped differential attention behavior (non-flash path), add integration tests (smoke inference). (Files: `src/models/motif.cpp`, llama-model.cpp, `src/llama-graph.*`, `tests/test-motif-attn.cpp`) 🔧🔥
6. Add GPU Flash / Vulkan / optimized kernels leveraging existing GQA and flash-attn infrastructure. Add shader/kernel unit tests. (Files: `ggml/src/...` + tests.) 🔥
7. Documentation & examples: conversion example, notes in docs, CI tests. ✅

---

## ✅ Tests to add (small list)
- `gguf-py/tests/test_motif_conversion.py` — ensure converter maps tensors & writes motif metadata.
- `tests/test-poly-norm.cpp` — verify PolyNorm matches NumPy/PyTorch reference.
- `tests/test-motif-attn.cpp` — CPU smoke test for grouped differential attention; verify shapes & numerical invariants.
- Extend flash/GQA tests to check grouped_ratio behavior.

---

## ⏱ Estimated effort (revised)
- Converter + gguf keys + tests: 0.5–1 day
- PolyNorm (CPU + tests): 1–2 days
- Motif build graph + CPU attention path + unit tests: 2–4 days
- GPU/Flash kernels & shaders + validation: 4–7 days
- Docs & CI: 1 day
Total realistic estimate: ~2–3 weeks (depending on GPU backend effort).

---

## Recommendation (next practical step)
Add the converter changes and gguf constants first (quick, low-risk, unblocks model round-trips), plus tests for conversion and a PolyNorm unit test. That gives us early validation while we design the attention kernel.

---

Would you like me to draft the exact minimal changes (file list + code locations) for the first PR (converter + gguf constants + conversion tests)? I can deliver a focused patch-plan and test skeleton next. 💡

— GitHub Copilot (using Raptor mini (Preview))

User: Please update the plan given this new information

GitHub Copilot: ## Updated Plan: MotifForCausalLM Support in llama.cpp

Based on repository scan findings, here's the revised implementation plan with adjusted priorities and realistic estimates.

---

### **Architecture Overview (unchanged)**
**Motif** uses Grouped Differential Attention with:
- Learnable lambda parameters for differential weighting
- PolyNorm activation (custom polynomial normalization)
- Sub-layer normalization in attention blocks
- Grouped head structure with `num_noise_heads` and `grouped_ratio`

---

### **What Already Exists (Reusable Infrastructure) ✅**

1. **Flash Attention & GQA Support**: Complete flash attention + grouped-query kernels in CUDA/Vulkan/OpenCL
2. **RoPE Infrastructure**: Rope helpers, freqs tensors, scaling metadata fully implemented
3. **RMSNorm Primitives**: T5LayerNorm/RMSNorm available across backends
4. **Converter Pattern**: `@ModelBase.register` + `modify_tensors` hooks work well
5. **GGUF Writer**: Generic `add_key_value` + typed helpers ready to use
6. **Model Layer Structure**: `llama_layer` has slots for `attn_sub_norm`, `ffn_sub_norm`
7. **Test Infrastructure**: Backend ops tests with GQA/flash attention scaffolding
8. **Architecture Examples**: Apertus, Dream, LLaDA provide patterns to follow

---

### **What Needs Implementation (Gaps) ⚠️**

1. **No Motif-specific code**: Zero mentions of "motif", "MotifForCausalLM", "poly_norm", "lambda_q/k"
2. **No PolyNorm**: Missing activation primitive/composed op
3. **No Differential Attention Kernel**: Core grouped differential attention logic
4. **No GGUF Keys**: Missing `num_noise_heads`, `grouped_ratio`, lambda tensor mappings
5. **No Converter Class**: Need `MotifModel(TextModel)` implementation
6. **No Tests**: Need conversion, PolyNorm, and attention unit tests

---

### **Revised Implementation Plan (6 Phases)**

#### **Phase 1: Foundation (Converter + GGUF) — 1 day** 🟢 *Low Risk*

**Goal**: Enable model conversion and metadata storage

**Files to modify**:
- constants.py
  - Add `MOTIF = auto()` to `MODEL_ARCH`
  - Add `MODEL_ARCH.MOTIF: "motif"` to `MODEL_ARCH_NAMES`
  - Add tensor keys: `ATTN_LAMBDA_Q1/K1/Q2/K2`, `ATTN_SUB_NORM`, `FFN_POLYNORM_W/B`
  - Add metadata keys: `ATTENTION.NUM_NOISE_HEADS`, `ATTENTION.GROUPED_RATIO`, `ATTENTION.K_RATIO`
  - Add `MODEL_TENSORS[MODEL_ARCH.MOTIF]` mapping

- tensor_mapping.py
  - Add mappings for `lambda_q1/k1/q2/k2`, `subln.weight`, `act_fn.weight/bias`

- convert_hf_to_gguf.py (lines ~3497+)
  - Add `@ModelBase.register("MotifForCausalLM")` class
  - Implement `set_vocab()` → use `_set_vocab_gpt2()`
  - Implement `set_gguf_parameters()` → write motif-specific metadata
  - Implement `modify_tensors()` → handle lambda params, subln, polynorm tensors
  - Implement `generate_extra_tensors()` if needed for RoPE

**Deliverable**: Can convert Motif HF models to GGUF format with correct metadata

---

#### **Phase 2: Conversion Tests — 0.5 days** 🟢 *Low Risk*

**Goal**: Verify converter correctness before C++ work

**New file**:
- `gguf-py/tests/test_motif_conversion.py`
  - Test metadata keys present
  - Test lambda tensor shapes
  - Test PolyNorm param shapes
  - Test tokenizer handling

**Deliverable**: Automated tests ensure conversion quality

---

#### **Phase 3: PolyNorm Implementation — 2 days** 🟡 *Medium Risk*

**Goal**: Add PolyNorm activation primitive

**PolyNorm formula** (from Motif code):
```
output = w[0] * norm(x³) + w[1] * norm(x²) + w[2] * norm(x) + bias
where norm(x) = x / sqrt(mean(x²) + eps)
```

**Implementation options**:
1. **Composed ops** (faster): Implement using existing ggml operations (mul, pow, mean, sqrt, add)
2. **New primitive** (cleaner): Add `ggml_poly_norm()` to ggml

**Files to modify**:
- ggml.h — declare `ggml_poly_norm()` if new primitive
- ggml.c — implement CPU version
- `ggml/src/ggml-cpu.c` — CPU backend implementation
- Optionally: GPU backends (CUDA/Metal/Vulkan) — can defer

**New test file**:
- `tests/test-poly-norm.cpp`
  - Unit test against NumPy reference
  - Test shape broadcasting
  - Test numerical accuracy

**Deliverable**: Working PolyNorm operation with CPU support

---

#### **Phase 4: Motif Architecture (C++) — 3 days** 🟡 *Medium Risk*

**Goal**: Add Motif model loading and basic CPU inference

**Files to modify**:

- llama-arch.cpp (add after line ~815)
  - Add `LLM_ARCH_MOTIF` case with tensor name mappings
  - Map lambda tensors, subln, polynorm weights

- llama-hparams.h (add after line ~165)
  - Add `uint32_t n_noise_heads = 0;`
  - Add `uint32_t n_grouped_ratio = 0;`
  - Add `uint32_t n_k_ratio = 1;`

- llama-model.cpp (add case ~line 6375)
  - Add `case LLM_ARCH_MOTIF:` in model loader
  - Load lambda tensors per layer (q1, k1, q2, k2, lambda_init)
  - Load subln weights per layer
  - Load polynorm weights/bias per layer
  - Load standard Q/K/V/O, FFN tensors

- llama-model.h (extend llama_layer ~line 200)
  - Add `ggml_tensor * lambda_q1/k1/q2/k2 = nullptr;`
  - Add `ggml_tensor * lambda_init = nullptr;`
  - Add `ggml_tensor * polynorm_w/b = nullptr;`
  - (Note: `attn_sub_norm` already exists)

**New file**:
- `src/models/motif.cpp` (follow llada.cpp pattern)
  - Implement `llm_build_motif::llm_build_motif()`
  - Build graph with standard embedding, RoPE
  - **Simplified attention** (initial): Use standard attention temporarily
  - Use PolyNorm in FFN gate projection
  - Add residual connections and norms

**Deliverable**: Can load Motif models and run basic inference (with simplified attention)

---

#### **Phase 5: Differential Attention Kernel — 4 days** 🔴 *High Risk*

**Goal**: Implement core Motif grouped differential attention

**Differential Attention Logic**:
```
1. Split Q into (Q1, Q2) by grouped_ratio
2. Split K/V similarly
3. Compute: attn1 = softmax(Q1·K1ᵀ/√d) · V1
4. Compute: attn2 = softmax(Q2·K2ᵀ/√d) · V2
5. Compute lambda: λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init
6. Output: subln((attn1 - λ·attn2) * (1-λ_init))
```

**Implementation approach**:
1. **CPU first**: Implement in `src/models/motif.cpp` using ggml ops
2. **Leverage GQA**: Reuse grouped-head logic from existing flash attention
3. **Add helper**: `build_differential_attn()` in graph builder

**Files to modify**:
- `src/models/motif.cpp`
  - Replace simplified attention with differential attention
  - Implement head grouping and splitting
  - Implement lambda computation
  - Implement attention subtraction and scaling
  - Apply subln after attention

- llama-graph.h (optional)
  - Add `build_attn_differential()` if generalizing

**GPU backends** (defer to Phase 6):
- Can initially run on CPU/eager path
- Flash attention optimization comes later

**Deliverable**: Correct Motif attention inference on CPU

---

#### **Phase 6: Tests & Optimization — 3 days** 🟡 *Medium Risk*

**Goal**: Add tests, examples, and GPU optimization

**Tasks**:

1. **Unit Tests**:
   - `tests/test-motif-attn.cpp` — smoke test attention outputs
   - Extend test-backend-ops.cpp — add GQA/grouped-ratio cases
   - Add perplexity test with small Motif model

2. **Conversion Example**:
   - `examples/model-conversion/scripts/causal/run-motif-model.py`
   - Demonstrate conversion workflow
   - Add validation checks

3. **GPU Optimization** (optional but recommended):
   - Adapt CUDA flash attention kernel for differential attention
   - Extend Vulkan/Metal shaders if time permits
   - Benchmark vs CPU

4. **Documentation**:
   - Update build.md with Motif support notes
   - Add conversion guide
   - Add architecture notes

**Deliverable**: Tested, documented Motif support with optional GPU acceleration

---

### **Timeline & Effort (Revised)**

| Phase | Description | Days | Risk |
|-------|-------------|------|------|
| 1 | Converter + GGUF keys | 1 | 🟢 Low |
| 2 | Conversion tests | 0.5 | 🟢 Low |
| 3 | PolyNorm implementation | 2 | 🟡 Medium |
| 4 | C++ model loading + basic graph | 3 | 🟡 Medium |
| 5 | Differential attention kernel | 4 | 🔴 High |
| 6 | Tests + GPU optimization | 3 | 🟡 Medium |
| **Total** | | **13.5 days** | **~2.5 weeks** |

---

### **Key Changes from Original Plan**

**What's easier now**:
- ✅ RoPE/flash/GQA infrastructure exists → saves ~3 days
- ✅ Converter pattern is proven → saves ~1 day
- ✅ Model layer structure ready → saves ~1 day

**What's still complex**:
- ⚠️ Differential attention is novel (no existing kernel)
- ⚠️ PolyNorm needs custom implementation
- ⚠️ GPU optimization requires kernel expertise

**Recommended approach**:
1. **Merge Phases 1-2 quickly** (converter + tests) — unblocks testing
2. **Implement Phase 3-4 in parallel** — PolyNorm + model structure
3. **Phase 5 is critical path** — focus here, accept CPU-only initially
4. **Phase 6 can be incremental** — tests first, GPU later

---

### **Immediate Next Steps (To Start Today)**

1. Add `MODEL_ARCH.MOTIF` to constants.py (5 min)
2. Add tensor key enums (15 min)
3. Implement `MotifModel` converter class (2 hours)
4. Add basic conversion test (30 min)
5. Test converter with Motif-2-12.7B model (1 hour)

This gives us working conversion by end of day 1, enabling all downstream work.

User: Am I going to need the whole file downloaded locally to do the conversion? Can I outsource that to cloud compute? Also what test file / pattern are you emulating to write the test which confirms that conversion worked properly? How will you know if we converted it properly

GitHub Copilot: - You do NOT need to have the entire model file checked into the repo. The converter (convert_hf_to_gguf.py) can download from the Hugging Face hub at runtime (or you can run the converter on a cloud VM and point it at the HF repo id). You do need enough disk to store the downloaded weight files during conversion (no streaming partial-conversion supported today). For very large models, use a cloud VM with sufficient disk and network — that’s the recommended approach.

- Test pattern to emulate: follow test_metadata.py (unittest style). Test should:
  1. run the converter (CLI or callable API) on a small Motif HF repo id or a small fixture checkpoint (run in a tempdir),
  2. load the produced .gguf using the gguf Python loader,
  3. assert architecture metadata == "motif" and motif-specific metadata keys exist,
  4. assert required tensor names exist and have expected shapes (e.g. lambda_q1/k1/q2/k2 per-layer shapes).

- How to know conversion is correct (checks to automate, increasing confidence):
  1. Metadata present: gguf.arch == "motif" and gguf.metadata keys (num_noise_heads, grouped_ratio, rope_theta, etc).
  2. Tensor presence & shapes: each motif tensor mapped (attn.lambda_q1/k1/q2/k2, attn.subln, ffn.polynorm_w/b, q/k/v/wo, etc) and shapes match HF config.
  3. Spot-check numeric values: compare a few tensor values in the gguf file vs the original HF tensors (via hf_hub or local checkpoint) to catch mapping/transpose errors.
  4. Functional smoke test: run a short inference with llama.cpp on the converted model and with reference HF implementation (same prompt) and compare logits or first-token probabilities within a small tolerance.
  5. Larger validation: run perplexity or end-to-end generation on a couple prompts and compare qualitative/quantitative results.

Minimal test skeleton (unittest) to emulate:

````python
#!/usr/bin/env python3
import unittest
import tempfile
import subprocess
from pathlib import Path
import gguf

class TestMotifConversion(unittest.TestCase):
    def test_convert_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "motif.gguf"
            # run converter; replace HF_ID with a small Motif test repo or local checkpoint
            subprocess.check_call([
                "python3", "convert_hf_to_gguf.py",
                "--model", "hf-org/Motif-small-test",  # <-- use a small test model or fixture
                "--out", str(out)
            ])
            g = gguf.load(str(out))
            self.assertEqual(g.model_arch, gguf.MODEL_ARCH.MOTIF)
            # motif-specific keys
            self.assertIn("motif.attention.num_noise_heads", g.metadata)
            self.assertIn("motif.attn.lambda_q1.weight.0", g.tensors)  # example tensor key
            # check shape of a tensor
            self.assertEqual(g.tensors["motif.attn.lambda_q1.weight.0"].shape[0], expected_head_dim)

if __name__ == "__main__":
    unittest.main()
````

If you want, I can:
- draft the exact converter changes and gguf constants,
- produce the concrete test file (with exact tensor keys/names derived from HF mapping),
- or prepare a small HF fixture repo id to use in CI/cloud.

Would you like me to generate the test file now?

User: no no, I'm asking you to find in the repo an example test which tests the conversion from hf to gguf (that you are going to try to emulate when implementing this feature)

GitHub Copilot: Summarized conversation historyCreating a short TODO list and then searching the repo for existing tests that exercise HF→GGUF conversion so I can emulate their pattern.

Created 3 todos

Summarized conversation history

Searched for text `convert_hf_to_gguf`, 50 results

Summarized conversation historyI'll search the repository for tests referencing the converter or GGUF conversion to find a usable example.

Searched for text `convert_hf_to_gguf.py` (`**/tests/**`), 2 results

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/tests/test-lora-conversion-inference.sh)

Searched for files matching `**/gguf-py/tests`, no matches

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/gguf-py)

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/gguf-py/tests)

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/gguf-py/tests/test_metadata.py)

Read [](file:///Users/pawanpaleja/VScodeProjects/llama.cpp/gguf-py/tests/test_quants.py)

**Conversion Test Example**: test-lora-conversion-inference.sh — shell integration test that runs convert_hf_to_gguf.py, convert_lora_to_gguf.py, and does end-to-end inference checks.

**Related gguf unit tests**:
- **Metadata**: test_metadata.py — unittest style example for gguf API.
- **Quants**: test_quants.py — heavier numeric test showing how to import local `gguf` package.

Next step: do you want me to
- draft a new Python unit test that emulates the gguf-py style and exercises convert_hf_to_gguf.py for Motif (fast, isolated GGUF-key checks), or
- update convert_hf_to_gguf.py to add a `MotifModel` skeleton now?
