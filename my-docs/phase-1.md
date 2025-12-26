# Phase 1: Preparation & Analysis - Motif-2-12.7B-Reasoning

## Tokenizer & Model Inspection

- [x] **Identify tokenizer type used by MotifForCausalLM**
  - **Finding:** The model uses a BPE-based tokenizer consistent with Llama-3 / Tiktoken.
  - **Evidence:** `tokenizer_config.json` and `tokenizer.json` are present in the Hugging Face repository.
  - **Type:** `LlamaTokenizerFast` / `PreTrainedTokenizerFast`.

- [x] **Check if tokenizer is already supported in llama.cpp**
  - **Finding:** Yes, supported.
  - **Action:** `convert_hf_to_gguf.py` supports the class, but we must map the `MotifForCausalLM` architecture to a new GGUF model type (e.g., `motif`) because the graph structure differs significantly from `llama`.

- [x] **Download `model.safetensors.index.json` and `config.json` from reference model**
  - **Status:** **Done**. Verified against file content.
  - **Config Highlights:**
    - `architectures`: `["MotifForCausalLM"]`
    - `hidden_act`: `poly_norm`
    - `rope_theta`: 1000000

- [x] **Create checklist of all model tensors in `model.safetensors.index.json`**
  - **Definitive Tensor List (Verified):**
    - **Global:**
      - `model.embed_tokens.weight`
      - `model.norm.weight`
      - `lm_head.weight`
    - **Per Layer (0-39):**
      - **Attention (GDA):**
        - `model.layers.{i}.self_attn.q_proj.weight`
        - `model.layers.{i}.self_attn.k_proj.weight`
        - `model.layers.{i}.self_attn.v_proj.weight`
        - `model.layers.{i}.self_attn.o_proj.weight`
        - **New/Unique:** `model.layers.{i}.self_attn.lambda_q1` (Learnable scalar/vector)
        - **New/Unique:** `model.layers.{i}.self_attn.lambda_q2`
        - **New/Unique:** `model.layers.{i}.self_attn.lambda_k1`
        - **New/Unique:** `model.layers.{i}.self_attn.lambda_k2`
        - **New/Unique:** `model.layers.{i}.self_attn.subln.weight` (Likely "Sub-LayerNorm" inside attention)
      - **MLP (PolyNorm):**
        - `model.layers.{i}.mlp.gate_proj.weight`
        - `model.layers.{i}.mlp.up_proj.weight`
        - `model.layers.{i}.mlp.down_proj.weight`
        - **New/Unique:** `model.layers.{i}.mlp.act_fn.weight` (PolyNorm learnable weight)
        - **New/Unique:** `model.layers.{i}.mlp.act_fn.bias` (PolyNorm learnable bias)
      - **Norms:**
        - `model.layers.{i}.input_layernorm.weight`
        - `model.layers.{i}.post_attention_layernorm.weight`

- [x] **Review `modeling_motif.py` to understand layer structure and parameters**
  - **Key Classes:** `MotifAttention` (uses the lambdas), `PolyNorm` (uses act_fn weights).
  - **Structure:** 40 Layers. The `lambda` parameters are critical for the differential attention calculation. `subln` suggests a normalization step likely applied to Q/K before attention.

- [x] **Review vLLM reference for additional implementation details**
  - **Finding:** Uses `DIFFERENTIAL_FLASH_ATTN`.
  - **Insight:** We must implement the graph to load and utilize the `lambda` tensors during the attention computation.

---

## Architecture Decisions

- [x] **Identify if GroupedDifferentialAttention can use existing GGML attention primitives**
  - **Decision:** **NO**.
  - **Reasoning:** Standard attention does not support the subtraction of two attention maps weighted by learnable `lambda` parameters (`lambda_q1`, etc.).
  - **Plan:**
    - Load `lambda` tensors in `llama_model_loader`.
    - Implement graph: `Attn = (Softmax(Q1*K1) - (lambda * Softmax(Q2*K2))) * V`.
    - Handle `subln` (Sub-LayerNorm) which likely normalizes the queries/keys.

- [x] **Identify if PolyNorm can be implemented as a variant of existing layer norm**
  - **Decision:** **NO**.
  - **Reasoning:** The presence of `mlp.act_fn.weight` and `bias` confirms this is a learnable, parameter-heavy activation, not a static function like SiLU.
  - **Plan:** Implement `ggml_polynorm` or a composite block that applies: `x = x * (weight * x + bias)` (or similar polynomial expansion found in `modeling_motif.py`).

- [x] **Determine if any expert/specialized tensor packing is needed**
  - **Decision:** **YES**.
  - **Detail:** We need to ensure the `lambda` scalars/vectors are packed correctly in the GGUF file. They are small, so they can be separate tensors or packed into the layer struct.

- [x] **Identify RoPE type (normal vs NeoX) from `apply_rotary_pos_emb` in reference code**
  - **Decision:** **Standard RoPE** (High Theta).