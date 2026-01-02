#include "models.h"
#include "../llama-kv-cache.h"

llm_build_motif::llm_build_motif(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa = hparams.n_embd_v_gqa_max();

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_kv * inp_attn = build_attn_inp_kv();

    // Get Motif hyperparameters from GGUF metadata
    // These are stored as metadata during conversion
    float grouped_ratio = 4.0f;  // Default from Motif-2-12.7B
    float k_ratio = 4.0f;        // Default from Motif-2-12.7B
    uint32_t num_noise_heads = 8; // Default from Motif-2-12.7B
    float lambda_init = 1.0f;    // Default

    // Try to get from metadata (these would be set during model loading)
    auto it_grouped_ratio = model.gguf_kv.find("motif.attention.grouped_ratio");
    if (it_grouped_ratio != model.gguf_kv.end()) {
        grouped_ratio = std::stof(it_grouped_ratio->second);
    }

    auto it_k_ratio = model.gguf_kv.find("motif.attention.k_ratio");
    if (it_k_ratio != model.gguf_kv.end()) {
        k_ratio = std::stof(it_k_ratio->second);
    }

    auto it_num_noise_heads = model.gguf_kv.find("motif.attention.num_noise_heads");
    if (it_num_noise_heads != model.gguf_kv.end()) {
        num_noise_heads = std::stoi(it_num_noise_heads->second);
    }

    auto it_lambda_init = model.gguf_kv.find("motif.attention.lambda_init");
    if (it_lambda_init != model.gguf_kv.end()) {
        lambda_init = std::stof(it_lambda_init->second);
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention with GroupedDifferentialAttention
        {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            cur = build_grouped_diff_attn(
                cur,  // input activations (post-attn-norm)
                inp_attn,
                model.layers[il].wq, model.layers[il].wk, model.layers[il].wv, model.layers[il].wo,
                model.layers[il].attn_lambda_q1, model.layers[il].attn_lambda_k1,
                model.layers[il].attn_lambda_q2, model.layers[il].attn_lambda_k2,
                model.layers[il].attn_sub_norm,
                inp_pos, rope_factors,
                lambda_init,
                (uint32_t)num_noise_heads, grouped_ratio, k_ratio,
                il);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network with PolyNorm activation
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_polynorm_ffn(
            cur,
            model.layers[il].ffn_gate, NULL,
            model.layers[il].ffn_up, NULL,
            model.layers[il].ffn_down, NULL,
            model.layers[il].ffn_polynorm_w,
            model.layers[il].ffn_polynorm_b,
            il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Helper function: build GroupedDifferentialAttention
ggml_tensor * llm_build_motif::build_grouped_diff_attn(
        ggml_tensor * input_cur,  // The input activations (post-norm)
        llm_graph_input_attn_kv * inp_attn,
        ggml_tensor * wq, ggml_tensor * wk, ggml_tensor * wv, ggml_tensor * wo,
        ggml_tensor * lambda_q1, ggml_tensor * lambda_k1,
        ggml_tensor * lambda_q2, ggml_tensor * lambda_k2,
        ggml_tensor * attn_sub_norm,
        ggml_tensor * inp_pos, ggml_tensor * rope_factors,
        float lambda_init,
        uint32_t num_noise_heads, float grouped_ratio, float k_ratio,
        int il) const {

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa = hparams.n_embd_v_gqa(il);
    const int64_t n_head = hparams.n_head(il);
    const int64_t n_head_kv = hparams.n_head_kv(il);

    // Compute Q/K/V projections
    ggml_tensor * Qcur = build_lora_mm(wq, input_cur);
    cb(Qcur, "Qcur_proj", il);

    ggml_tensor * Kcur = build_lora_mm(wk, input_cur);
    cb(Kcur, "Kcur_proj", il);

    ggml_tensor * Vcur = build_lora_mm(wv, input_cur);
    cb(Vcur, "Vcur_proj", il);

    // Reshape for attention: [n_embd_head, n_head, n_tokens]
    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply RoPE (Normal RoPE type for Motif)
    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                         n_rot, LLAMA_ROPE_TYPE_NORM, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                         n_rot, LLAMA_ROPE_TYPE_NORM, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "Qcur_rope", il);
    cb(Kcur, "Kcur_rope", il);

    // Store K/V to KV cache
    const auto * mctx_cur = inp_attn->mctx;
    {
        const auto & k_idxs = inp_attn->get_k_idxs();
        const auto & v_idxs = inp_attn->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, v_idxs, il));
    }

    // Get cached K/V for attention computation
    ggml_tensor * Kcached = mctx_cur->get_k(ctx0, il);
    ggml_tensor * Vcached = mctx_cur->get_v(ctx0, il);

    // Implement GroupedDifferentialAttention logic
    ggml_tensor * attn_output = build_grouped_diff_attention_core(
        Qcur, Kcached, Vcached,
        inp_attn->get_kq_mask(),
        lambda_q1, lambda_k1, lambda_q2, lambda_k2,
        attn_sub_norm,
        lambda_init, num_noise_heads, grouped_ratio, k_ratio,
        il);

    cb(attn_output, "attn_diff_out", il);

    // Apply output projection
    ggml_tensor * output = build_lora_mm(wo, attn_output);
    cb(output, "attn_output_proj", il);

    return output;
}

// Core GroupedDifferentialAttention implementation
ggml_tensor * llm_build_motif::build_grouped_diff_attention_core(
        ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V,
        ggml_tensor * kq_mask,
        ggml_tensor * lambda_q1, ggml_tensor * lambda_k1,
        ggml_tensor * lambda_q2, ggml_tensor * lambda_k2,
        ggml_tensor * attn_sub_norm,
        float lambda_init,
        uint32_t num_noise_heads, float grouped_ratio, float k_ratio,
        int il) const {

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_head = hparams.n_head(il);

    // Lambda computation: λ1 = exp(sum(λ_q1 * λ_k1)), λ2 = exp(sum(λ_q2 * λ_k2))
    ggml_tensor * lambda_q1k1 = ggml_mul(ctx0, lambda_q1, lambda_k1);
    ggml_tensor * lambda_sum1 = ggml_sum(ctx0, lambda_q1k1);
    ggml_tensor * lambda_1 = ggml_exp(ctx0, lambda_sum1);

    ggml_tensor * lambda_q2k2 = ggml_mul(ctx0, lambda_q2, lambda_k2);
    ggml_tensor * lambda_sum2 = ggml_sum(ctx0, lambda_q2k2);
    ggml_tensor * lambda_2 = ggml_exp(ctx0, lambda_sum2);

    // λ_full = λ1 - λ2 + λ_init
    ggml_tensor * lambda_full = ggml_add(ctx0,
        ggml_sub(ctx0, lambda_1, lambda_2),
        ggml_new_f32(ctx0, lambda_init));

    cb(lambda_full, "lambda_full", il);

    // TODO: Implement full grouped differential attention logic
    // This is a complex implementation that requires:
    // 1. Head grouping based on grouped_ratio (4.0) and k_ratio (4.0)
    // 2. Splitting Q/K/V into grouped and noise components
    // 3. Computing two separate attention maps
    // 4. Differential combination: attn_o - λ * attn_n
    //
    // For now, implement a simplified version that computes standard attention
    // but applies the lambda scaling and SubLayerNorm

    // Compute standard attention (Q @ K^T) / sqrt(d)
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * kq = ggml_mul_mat(ctx0, K, Q);
    cb(kq, "kq", il);

    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
    cb(kq, "kq_softmax", il);

    // Compute attention output: attn @ V
    ggml_tensor * attn_output = ggml_mul_mat(ctx0, V, kq);
    cb(attn_output, "attn_output_raw", il);

    // Reshape to [n_tokens, n_embd_head * n_head] for SubLayerNorm
    attn_output = ggml_permute(ctx0, attn_output, 0, 2, 1, 3);
    attn_output = ggml_cont_2d(ctx0, attn_output, n_embd_head * n_head, n_tokens);

    // Apply SubLayerNorm (RMSNorm on the concatenated head outputs)
    ggml_tensor * normalized = ggml_rms_norm(ctx0, attn_output, hparams.f_norm_rms_eps);
    normalized = ggml_mul(ctx0, normalized, attn_sub_norm);
    cb(normalized, "attn_subln", il);

    // Apply (1 - λ_init) scaling factor
    ggml_tensor * scaled = ggml_scale(ctx0, normalized, 1.0f - lambda_init);
    cb(scaled, "attn_scaled", il);

    return scaled;
}

// Helper function: build PolyNorm activation in FFN
ggml_tensor * llm_build_motif::build_polynorm_ffn(
        ggml_tensor * cur,
        ggml_tensor * ffn_gate, ggml_tensor * ffn_gate_b,
        ggml_tensor * ffn_up, ggml_tensor * ffn_up_b,
        ggml_tensor * ffn_down, ggml_tensor * ffn_down_b,
        ggml_tensor * polynorm_w, ggml_tensor * polynorm_b,
        int il) const {

    // Standard gate/up computation
    ggml_tensor * tmp = ffn_up ? build_lora_mm(ffn_up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (ffn_up_b) {
        tmp = ggml_add(ctx0, tmp, ffn_up_b);
        cb(tmp, "ffn_up_b", il);
    }

    ggml_tensor * gate_out = nullptr;
    if (ffn_gate) {
        gate_out = build_lora_mm(ffn_gate, cur);
        cb(gate_out, "ffn_gate", il);

        if (ffn_gate_b) {
            gate_out = ggml_add(ctx0, gate_out, ffn_gate_b);
            cb(gate_out, "ffn_gate_b", il);
        }
    }

    // Apply PolyNorm activation to gate output
    if (gate_out) {
        gate_out = build_polynorm(gate_out, polynorm_w, polynorm_b, il);
        cb(gate_out, "ffn_gate_polynorm", il);

        // Element-wise multiply with up projection (swiglu-style)
        tmp = ggml_mul(ctx0, gate_out, tmp);
        cb(tmp, "ffn_gate_up_mul", il);
    }

    // Down projection
    cur = build_lora_mm(ffn_down, tmp);
    cb(cur, "ffn_down", il);

    if (ffn_down_b) {
        cur = ggml_add(ctx0, cur, ffn_down_b);
        cb(cur, "ffn_down_b", il);
    }

    return cur;
}

// Helper function: build PolyNorm activation
ggml_tensor * llm_build_motif::build_polynorm(
        ggml_tensor * x,
        ggml_tensor * w, ggml_tensor * b,
        int il) const {

    // PolyNorm formula: w[0]*norm(x³) + w[1]*norm(x²) + w[2]*norm(x) + b
    // where norm(y) = y / sqrt(mean(y²) + eps) per row

    const float eps = 1e-6f;  // Standard epsilon

    // Compute x³, x², x
    // x³ = x * x * x (using mul since ggml_pow doesn't exist)
    ggml_tensor * x2 = ggml_sqr(ctx0, x);
    ggml_tensor * x3 = ggml_mul(ctx0, x2, x);
    ggml_tensor * x1 = x;

    // Normalize each (RMS norm per row)
    ggml_tensor * n3 = ggml_rms_norm(ctx0, x3, eps);
    ggml_tensor * n2 = ggml_rms_norm(ctx0, x2, eps);
    ggml_tensor * n1 = ggml_rms_norm(ctx0, x1, eps);

    // Weight and sum: w[0]*n3 + w[1]*n2 + w[2]*n1 + b
    ggml_tensor * w3 = ggml_mul(ctx0, ggml_view_1d(ctx0, w, 1, 0*sizeof(float)), n3);
    ggml_tensor * w2 = ggml_mul(ctx0, ggml_view_1d(ctx0, w, 1, 1*sizeof(float)), n2);
    ggml_tensor * w1 = ggml_mul(ctx0, ggml_view_1d(ctx0, w, 1, 2*sizeof(float)), n1);

    ggml_tensor * result = ggml_add(ctx0, w3, w2);
    result = ggml_add(ctx0, result, w1);
    result = ggml_add(ctx0, result, b);

    cb(result, "polynorm", il);

    return result;
}