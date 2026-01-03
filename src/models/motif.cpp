#include "../llama-kv-cache.h"
#include "models.h"

llm_build_motif::llm_build_motif(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_kv * inp_attn = build_attn_inp_kv();

    // Get Motif hyperparameters from GGUF metadata
    float    grouped_ratio   = 4.0f;  // Default from Motif-2-12.7B
    float    k_ratio         = 1.0f;  // Default
    uint32_t num_noise_heads = 8;     // Default from Motif-2-12.7B
    float    lambda_init     = 1.0f;  // Default

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
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention with GroupedDifferentialAttention
        {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            cur = build_grouped_diff_attn(cur, inp_attn, model.layers[il].wq, model.layers[il].wk, model.layers[il].wv,
                                          model.layers[il].wo, model.layers[il].attn_lambda_q1,
                                          model.layers[il].attn_lambda_k1, model.layers[il].attn_lambda_q2,
                                          model.layers[il].attn_lambda_k2, model.layers[il].attn_sub_norm, inp_pos,
                                          rope_factors, lambda_init, num_noise_heads, grouped_ratio, k_ratio, il);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_polynorm_ffn(cur, model.layers[il].ffn_gate, NULL, model.layers[il].ffn_up, NULL,
                                 model.layers[il].ffn_down, NULL, model.layers[il].ffn_polynorm_w,
                                 model.layers[il].ffn_polynorm_b, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Helper function: build GroupedDifferentialAttention
ggml_tensor * llm_build_motif::build_grouped_diff_attn(ggml_tensor *             input_cur,
                                                       llm_graph_input_attn_kv * inp_attn,
                                                       ggml_tensor *             wq,
                                                       ggml_tensor *             wk,
                                                       ggml_tensor *             wv,
                                                       ggml_tensor *             wo,
                                                       ggml_tensor *             lambda_q1,
                                                       ggml_tensor *             lambda_k1,
                                                       ggml_tensor *             lambda_q2,
                                                       ggml_tensor *             lambda_k2,
                                                       ggml_tensor *             attn_sub_norm,
                                                       ggml_tensor *             inp_pos,
                                                       ggml_tensor *             rope_factors,
                                                       float                     lambda_init,
                                                       uint32_t                  num_noise_heads,
                                                       float                     grouped_ratio,
                                                       float                     k_ratio,
                                                       int                       il) const {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_head      = hparams.n_head(il);
    const int64_t n_head_kv   = hparams.n_head_kv(il);

    // Compute Q/K/V projections
    ggml_tensor * Qcur = build_lora_mm(wq, input_cur);
    ggml_tensor * Kcur = build_lora_mm(wk, input_cur);
    ggml_tensor * Vcur = build_lora_mm(wv, input_cur);

    cb(Qcur, "Qcur_proj", il);
    cb(Kcur, "Kcur_proj", il);
    cb(Vcur, "Vcur_proj", il);

    // Reshape for RoPE: [n_embd_head, n_head, n_tokens]
    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply RoPE
    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot, LLAMA_ROPE_TYPE_NORM, n_ctx_orig, freq_base,
                         freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot, LLAMA_ROPE_TYPE_NORM, n_ctx_orig, freq_base,
                         freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "Qcur_rope", il);
    cb(Kcur, "Kcur_rope", il);

    // Store K/V to cache
    const auto * mctx_cur = inp_attn->mctx;
    {
        const auto & k_idxs = inp_attn->get_k_idxs();
        const auto & v_idxs = inp_attn->get_v_idxs();
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, v_idxs, il));
    }

    // Get cached K/V
    ggml_tensor * Kcached = mctx_cur->get_k(ctx0, il);
    ggml_tensor * Vcached = mctx_cur->get_v(ctx0, il);

    // Use standard llama.cpp attention helper which handles all the details
    // For Motif, we need to bypass the standard attention and implement GDA manually
    ggml_tensor * attn_output = build_grouped_diff_attention_core(
        Qcur, Kcached, Vcached, inp_attn->get_kq_mask(), lambda_q1, lambda_k1, lambda_q2, lambda_k2, attn_sub_norm,
        lambda_init, num_noise_heads, grouped_ratio, k_ratio, il);

    cb(attn_output, "attn_diff_out", il);

    // Apply output projection
    ggml_tensor * output = build_lora_mm(wo, attn_output);
    cb(output, "attn_output_proj", il);

    return output;
}

// Core GroupedDifferentialAttention implementation following vLLM exactly
// vLLM Reference: grouped_diff_attn.py forward() method
//
// Algorithm:
// 1. Split Q into Q1 (32 original heads) and Q2 (8 noise heads)
// 2. Split K/V into halves: K1/V1 and K2/V2 (8 heads each)
// 3. Compute 4 attention outputs:
//    - attn11 = attention(Q1, K1, V1) -> [N, 32, d]
//    - attn12 = attention(Q1, K1, V2) -> [N, 32, d]
//    - attn21 = attention(Q2, K2, V1) -> [N, 8, d]
//    - attn22 = attention(Q2, K2, V2) -> [N, 8, d]
// 4. Concatenate: attn1 = cat([attn11, attn12], dim=-1) -> [N, 32, 2d]
//                 attn2 = cat([attn21, attn22], dim=-1) -> [N, 8, 2d]
// 5. Repeat attn2 by grouped_ratio=4: -> [N, 32, 2d]
// 6. Differential: attn = attn1 - lambda_full * attn2
// 7. SubLayerNorm: attn = subln(attn)
// 8. Scale: attn = attn * (1 - lambda_init)
// 9. Reshape: [N, 32, 2d] -> [N, 64, d] for o_proj
ggml_tensor * llm_build_motif::build_grouped_diff_attention_core(ggml_tensor * Q,
                                                                 ggml_tensor * K,
                                                                 ggml_tensor * V,
                                                                 ggml_tensor * kq_mask,
                                                                 ggml_tensor * lambda_q1,
                                                                 ggml_tensor * lambda_k1,
                                                                 ggml_tensor * lambda_q2,
                                                                 ggml_tensor * lambda_k2,
                                                                 ggml_tensor * attn_sub_norm,
                                                                 float         lambda_init,
                                                                 uint32_t      num_noise_heads,
                                                                 float         grouped_ratio,
                                                                 float         k_ratio,
                                                                 int           il) const {
    // Silence unused params
    (void) grouped_ratio;
    (void) k_ratio;

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_head      = hparams.n_head(il);     // Total Q heads (40)
    const int64_t n_head_kv   = hparams.n_head_kv(il);  // Total KV heads (16)
    const int64_t n_kv        = K->ne[2];               // KV cache length

    // Q: [d, n_head, n_tokens] after RoPE
    // K: [d, n_head_kv, n_kv]
    // V: [d, n_head_kv, n_kv]

    // ========================================
    // 1. Calculate Lambda
    // ========================================
    // λ_full = exp(sum(λ_q1 * λ_k1)) - exp(sum(λ_q2 * λ_k2)) + λ_init
    ggml_tensor * lambda_q1k1 = ggml_mul(ctx0, lambda_q1, lambda_k1);
    ggml_tensor * lambda_sum1 = ggml_sum(ctx0, lambda_q1k1);
    ggml_tensor * lambda_1    = ggml_exp(ctx0, lambda_sum1);

    ggml_tensor * lambda_q2k2 = ggml_mul(ctx0, lambda_q2, lambda_k2);
    ggml_tensor * lambda_sum2 = ggml_sum(ctx0, lambda_q2k2);
    ggml_tensor * lambda_2    = ggml_exp(ctx0, lambda_sum2);

    ggml_tensor * lambda_diff = ggml_sub(ctx0, lambda_1, lambda_2);
    // lambda_full = lambda_diff + lambda_init (applied as scale later)

    // ========================================
    // 2. Split Q/K/V heads
    // ========================================
    // vLLM: Q splits into Q1 (original) and Q2 (noise)
    // vLLM: K/V split into interleaved halves: K1/V1 from even indices, K2/V2 from odd indices

    const int64_t n_head_q1  = n_head - num_noise_heads;  // 40 - 8 = 32
    const int64_t n_head_q2  = num_noise_heads;           // 8
    const int64_t n_head_kv1 = n_head_kv / 2;             // 16 / 2 = 8
    const int64_t n_head_kv2 = n_head_kv / 2;             // 8

    // Split Q: first n_head_q1 heads are Q1, remaining are Q2
    // Q is [d, n_head, n_tokens]
    ggml_tensor * Q1 = ggml_view_3d(ctx0, Q, n_embd_head, n_head_q1, n_tokens, Q->nb[1], Q->nb[2], 0);
    ggml_tensor * Q2 = ggml_view_3d(ctx0, Q, n_embd_head, n_head_q2, n_tokens, Q->nb[1], Q->nb[2],
                                    n_embd_head * n_head_q1 * ggml_element_size(Q));

    // Split K/V: first half are K1/V1, second half are K2/V2
    // K is [d, n_head_kv, n_kv]
    ggml_tensor * K1 = ggml_view_3d(ctx0, K, n_embd_head, n_head_kv1, n_kv, K->nb[1], K->nb[2], 0);
    ggml_tensor * K2 = ggml_view_3d(ctx0, K, n_embd_head, n_head_kv2, n_kv, K->nb[1], K->nb[2],
                                    n_embd_head * n_head_kv1 * ggml_element_size(K));

    ggml_tensor * V1 = ggml_view_3d(ctx0, V, n_embd_head, n_head_kv1, n_kv, V->nb[1], V->nb[2], 0);
    ggml_tensor * V2 = ggml_view_3d(ctx0, V, n_embd_head, n_head_kv2, n_kv, V->nb[1], V->nb[2],
                                    n_embd_head * n_head_kv1 * ggml_element_size(V));

    // ========================================
    // 3. Compute 4 attention outputs using build_attn_mha
    // ========================================
    // build_attn_mha handles the permutation and mul_mat internally
    // It expects Q: [d, H_q, N_q], K: [d, H_kv, N_kv], V: [d, H_kv, N_kv]
    // Returns: [d, N_q, H_q] (after permuting back)

    const float kq_scale =
        hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // attn11 = attention(Q1, K1, V1) -> [d, n_tokens, 32]
    ggml_tensor * attn11 = build_attn_mha(Q1, K1, V1, nullptr, kq_mask, nullptr, nullptr, kq_scale, il);
    cb(attn11, "attn11", il);

    // attn12 = attention(Q1, K1, V2) -> [d, n_tokens, 32]
    ggml_tensor * attn12 = build_attn_mha(Q1, K1, V2, nullptr, kq_mask, nullptr, nullptr, kq_scale, il);
    cb(attn12, "attn12", il);

    // attn21 = attention(Q2, K2, V1) -> [d, n_tokens, 8]
    ggml_tensor * attn21 = build_attn_mha(Q2, K2, V1, nullptr, kq_mask, nullptr, nullptr, kq_scale, il);
    cb(attn21, "attn21", il);

    // attn22 = attention(Q2, K2, V2) -> [d, n_tokens, 8]
    ggml_tensor * attn22 = build_attn_mha(Q2, K2, V2, nullptr, kq_mask, nullptr, nullptr, kq_scale, il);
    cb(attn22, "attn22", il);

    // ========================================
    // 4. Concatenate along head_dim: [d, N, H] -> [2d, N, H]
    // ========================================
    // attn1 = cat([attn11, attn12], dim=0) -> [2d, n_tokens, 32]
    ggml_tensor * attn1 = ggml_new_tensor_3d(ctx0, attn11->type, n_embd_head * 2, n_tokens, n_head_q1);

    // Copy attn11 to first half
    ggml_tensor * attn1_dst1 =
        ggml_view_3d(ctx0, attn1, n_embd_head, n_tokens, n_head_q1, attn1->nb[1], attn1->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, attn11, attn1_dst1));

    // Copy attn12 to second half
    ggml_tensor * attn1_dst2 = ggml_view_3d(ctx0, attn1, n_embd_head, n_tokens, n_head_q1, attn1->nb[1], attn1->nb[2],
                                            n_embd_head * ggml_element_size(attn1));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, attn12, attn1_dst2));

    // attn2 = cat([attn21, attn22], dim=0) -> [2d, n_tokens, 8]
    ggml_tensor * attn2 = ggml_new_tensor_3d(ctx0, attn21->type, n_embd_head * 2, n_tokens, n_head_q2);

    ggml_tensor * attn2_dst1 =
        ggml_view_3d(ctx0, attn2, n_embd_head, n_tokens, n_head_q2, attn2->nb[1], attn2->nb[2], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, attn21, attn2_dst1));

    ggml_tensor * attn2_dst2 = ggml_view_3d(ctx0, attn2, n_embd_head, n_tokens, n_head_q2, attn2->nb[1], attn2->nb[2],
                                            n_embd_head * ggml_element_size(attn2));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, attn22, attn2_dst2));

    // ========================================
    // 5. Repeat attn2 by gqa_ratio to match attn1 heads
    // ========================================
    // attn2: [2d, n_tokens, 8] -> [2d, n_tokens, 32]
    ggml_tensor * attn2_rep = ggml_repeat(ctx0, attn2, attn1);

    // ========================================
    // 6. Differential: attn = attn1 - lambda_full * attn2
    // ========================================
    // lambda_full = lambda_diff + lambda_init
    ggml_tensor * attn2_scaled_diff = ggml_mul(ctx0, attn2_rep, lambda_diff);
    ggml_tensor * attn2_scaled_init = ggml_scale(ctx0, attn2_rep, lambda_init);
    ggml_tensor * attn2_scaled      = ggml_add(ctx0, attn2_scaled_diff, attn2_scaled_init);

    ggml_tensor * attn_diff = ggml_sub(ctx0, attn1, attn2_scaled);

    // ========================================
    // 7. SubLayerNorm (RMSNorm on 2*d dimension)
    // ========================================
    ggml_tensor * attn_norm = ggml_rms_norm(ctx0, attn_diff, hparams.f_norm_rms_eps);
    attn_norm               = ggml_mul(ctx0, attn_norm, attn_sub_norm);

    // ========================================
    // 8. Scale: attn = attn * (1 - lambda_init)
    // ========================================
    ggml_tensor * attn_scaled = ggml_scale(ctx0, attn_norm, 1.0f - lambda_init);

    // ========================================
    // 9. Reshape for output projection
    // ========================================
    // [2d, n_tokens, 32] -> permute to [2d, 32, n_tokens] -> flatten to [2d*32, n_tokens]
    ggml_tensor * attn_permuted = ggml_permute(ctx0, attn_scaled, 0, 2, 1, 3);
    ggml_tensor * attn_flat     = ggml_cont_2d(ctx0, attn_permuted, n_embd_head * 2 * n_head_q1, n_tokens);

    return attn_flat;
}

// Helper function: build PolyNorm activation in FFN
ggml_tensor * llm_build_motif::build_polynorm_ffn(ggml_tensor * cur,
                                                  ggml_tensor * ffn_gate,
                                                  ggml_tensor * ffn_gate_b,
                                                  ggml_tensor * ffn_up,
                                                  ggml_tensor * ffn_up_b,
                                                  ggml_tensor * ffn_down,
                                                  ggml_tensor * ffn_down_b,
                                                  ggml_tensor * polynorm_w,
                                                  ggml_tensor * polynorm_b,
                                                  int           il) const {
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

    if (gate_out) {
        gate_out = build_polynorm(gate_out, polynorm_w, polynorm_b, il);
        cb(gate_out, "ffn_gate_polynorm", il);

        tmp = ggml_mul(ctx0, gate_out, tmp);
        cb(tmp, "ffn_gate_up_mul", il);
    }

    cur = build_lora_mm(ffn_down, tmp);
    cb(cur, "ffn_down", il);

    if (ffn_down_b) {
        cur = ggml_add(ctx0, cur, ffn_down_b);
        cb(cur, "ffn_down_b", il);
    }

    return cur;
}

// Helper function: build PolyNorm activation
// PolyNorm: output = w[0]*RMSNorm(x³) + w[1]*RMSNorm(x²) + w[2]*RMSNorm(x) + bias
// Reference: https://arxiv.org/html/2411.03884v1
ggml_tensor * llm_build_motif::build_polynorm(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int il) const {
    const float eps = 1e-6f;

    // x: [intermediate_size, n_tokens]
    // w: [3] - three learned weights
    // b: [1] - bias

    // Compute x^2 and x^3
    ggml_tensor * x2 = ggml_sqr(ctx0, x);
    ggml_tensor * x3 = ggml_mul(ctx0, x2, x);

    // RMSNorm each polynomial term
    // Note: ggml_rms_norm normalizes along dimension 0
    ggml_tensor * n3 = ggml_rms_norm(ctx0, x3, eps);
    ggml_tensor * n2 = ggml_rms_norm(ctx0, x2, eps);
    ggml_tensor * n1 = ggml_rms_norm(ctx0, x, eps);

    // For scalar multiplication, we need to extract scalar values and use ggml_scale
    // But ggml_scale only works with compile-time constants.
    // Instead, we reshape weight views to be broadcastable:
    // Create a [1, 1] tensor that can broadcast to [intermediate_size, n_tokens]

    // View each weight element as [1, 1] and repeat to match n3 shape
    // Actually simpler: just use ggml_scale with a proper broadcast
    // GGML requires b to have dimensions that divide a for repeating.

    // Use a different approach: construct the weighted sum manually
    // w0*n3: Use ggml_cont to make the scalar contiguous, then ggml_repeat

    // Create 1-element tensors for each weight
    ggml_tensor * w0   = ggml_view_1d(ctx0, w, 1, 0 * sizeof(float));
    ggml_tensor * w1_t = ggml_view_1d(ctx0, w, 1, 1 * sizeof(float));  // renamed to avoid conflict with w1
    ggml_tensor * w2_t = ggml_view_1d(ctx0, w, 1, 2 * sizeof(float));

    // Reshape to [1, 1] to match 2D tensor for broadcasting
    w0   = ggml_reshape_2d(ctx0, w0, 1, 1);
    w1_t = ggml_reshape_2d(ctx0, w1_t, 1, 1);
    w2_t = ggml_reshape_2d(ctx0, w2_t, 1, 1);

    // Repeat to match n3 shape [intermediate_size, n_tokens]
    ggml_tensor * w0_rep = ggml_repeat(ctx0, w0, n3);
    ggml_tensor * w1_rep = ggml_repeat(ctx0, w1_t, n2);
    ggml_tensor * w2_rep = ggml_repeat(ctx0, w2_t, n1);

    // Weighted terms
    ggml_tensor * term0 = ggml_mul(ctx0, w0_rep, n3);
    ggml_tensor * term1 = ggml_mul(ctx0, w1_rep, n2);
    ggml_tensor * term2 = ggml_mul(ctx0, w2_rep, n1);

    // Sum all terms
    ggml_tensor * result = ggml_add(ctx0, term0, term1);
    result               = ggml_add(ctx0, result, term2);

    // Add bias (b is [1], reshape to [1, 1] for broadcasting)
    if (b != nullptr) {
        ggml_tensor * b_2d  = ggml_reshape_2d(ctx0, b, 1, 1);
        ggml_tensor * b_rep = ggml_repeat(ctx0, b_2d, result);
        result              = ggml_add(ctx0, result, b_rep);
    }

    cb(result, "polynorm", il);

    return result;
}
