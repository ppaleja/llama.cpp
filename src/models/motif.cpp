#include "../llama-kv-cache.h"
#include "models.h"

#include <cmath>

// Helper: Implements torch.repeat_interleave along dimension 1 (heads dimension)
// Input: [d, heads, seq]  -> Output: [d, heads*n_rep, seq]
// repeat_interleave: [A B C D] with n_rep=4 -> [A A A A B B B B C C C C D D D D]
// (NOT tiling which would be [A B C D A B C D A B C D A B C D])
static ggml_tensor * repeat_interleave_heads(ggml_context * ctx, ggml_tensor * x, int n_rep) {
    if (n_rep == 1) {
        return x;
    }

    const int64_t d     = x->ne[0];  // head_dim
    const int64_t heads = x->ne[1];  // num heads
    const int64_t seq   = x->ne[2];  // sequence length

    // Step 1: Reshape [d, heads, seq] -> [d, 1, heads, seq]
    // Then repeat to [d, n_rep, heads, seq]
    // Then reshape to [d, heads*n_rep, seq]

    // But ggml_repeat tiles on the target shape dimension, so we need a different approach:
    // Reshape [d, heads, seq] to [d, heads, 1, seq]
    ggml_tensor * x_4d = ggml_reshape_4d(ctx, x, d, heads, 1, seq);

    // Create target tensor [d, heads, n_rep, seq]
    ggml_tensor * target = ggml_new_tensor_4d(ctx, x->type, d, heads, n_rep, seq);

    // Repeat along dim 2: [d, heads, 1, seq] -> [d, heads, n_rep, seq]
    ggml_tensor * repeated = ggml_repeat(ctx, x_4d, target);

    // Permute to [d, n_rep, heads, seq] (swap dims 1 and 2)
    ggml_tensor * permuted = ggml_permute(ctx, repeated, 0, 2, 1, 3);

    // Make contiguous and reshape to [d, heads*n_rep, seq]
    ggml_tensor * result = ggml_cont_3d(ctx, permuted, d, heads * n_rep, seq);

    return result;
}

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
    float    lambda_init     = 0.0f;  // Default to 0.0f (safe), 1.0f leads to zero output!

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

            // Calculate lambda_init per layer: 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))
            // HuggingFace: self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))
            const float layer_lambda_init = 0.8f - 0.6f * std::exp(-0.3f * (il - 1));

            cur = build_grouped_diff_attn(cur, inp_attn, model.layers[il].wq, model.layers[il].wk, model.layers[il].wv,
                                          model.layers[il].wo, model.layers[il].attn_lambda_q1,
                                          model.layers[il].attn_lambda_k1, model.layers[il].attn_lambda_q2,
                                          model.layers[il].attn_lambda_k2, model.layers[il].attn_sub_norm, inp_pos,
                                          rope_factors, layer_lambda_init, num_noise_heads, grouped_ratio, k_ratio, il);
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

#if 1
    std::fprintf(stderr, "KV CACHE DIAGNOSTICS (layer %d):\n", il);
    std::fprintf(stderr, "  Kcached (raw): [%lld, %lld, %lld]\n", (long long) Kcached->ne[0],
                 (long long) Kcached->ne[1], (long long) Kcached->ne[2]);
#endif

    // FIX: Slice to valid context length (n_past + n_tokens)
    // The KV cache returns the full buffer (e.g. 32768), but we must only attend to valid tokens
    const int32_t n_kv_valid = mctx_cur->get_n_kv();

    if (Kcached->ne[2] > n_kv_valid) {
        Kcached =
            ggml_view_3d(ctx0, Kcached, Kcached->ne[0], Kcached->ne[1], n_kv_valid, Kcached->nb[1], Kcached->nb[2], 0);

        Vcached =
            ggml_view_3d(ctx0, Vcached, Vcached->ne[0], Vcached->ne[1], n_kv_valid, Vcached->nb[1], Vcached->nb[2], 0);
    }

#if 1
    std::fprintf(stderr, "  Kcached (sliced): [%lld, %lld, %lld]\n", (long long) Kcached->ne[0],
                 (long long) Kcached->ne[1], (long long) Kcached->ne[2]);
    std::fprintf(stderr, "  n_tokens (Q): %lld\n", (long long) n_tokens);
    std::fprintf(stderr, "  mctx->get_n_kv(): %d\n", n_kv_valid);

    ggml_tensor * mask = inp_attn->get_kq_mask();
    if (mask) {
        std::fprintf(stderr, "  kq_mask: [%lld, %lld, %lld, %lld]\n", (long long) mask->ne[0], (long long) mask->ne[1],
                     (long long) mask->ne[2], (long long) mask->ne[3]);
        // DIAGNOSTIC: Print first few mask values at runtime
        // Note: mask->data is only available at execution time, not graph-build time
        // We'll print buffer status instead
        std::fprintf(stderr, "  kq_mask data ptr: %p, buffer: %p\n", (void *) mask->data, (void *) mask->buffer);
    } else {
        std::fprintf(stderr, "  kq_mask: NULL\n");
    }

    // DIAGNOSTIC: Print first few values of K cache to verify it contains proper data
    // Note: These are graph-build time checks - actual data will be set during execution
    std::fprintf(stderr, "  Kcached data ptr: %p, buffer: %p\n", (void *) Kcached->data, (void *) Kcached->buffer);
    std::fprintf(stderr, "  Qcur data ptr: %p, buffer: %p\n", (void *) Qcur->data, (void *) Qcur->buffer);
#endif

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

    const int64_t n_tokens = Q->ne[2];

    // Q splitting using grouped rearrangement (matching HuggingFace)
    // Q: 40 heads → 8 groups of 5 → Q1=first 4 per group (32), Q2=last 1 per group (8)
    const int q_group_size = (int) grouped_ratio + 1;  // 5
    const int num_groups_q = n_head / q_group_size;    // 8

    // Reshape Q from [d, 40, N] to [d, 5, 8, N]
    ggml_tensor * Q_4d = ggml_reshape_4d(ctx0, Q, n_embd_head, q_group_size, num_groups_q, n_tokens);

    // Q1 = first 4 heads from each group → 32 heads
    ggml_tensor * Q1_4d = ggml_view_4d(ctx0, Q_4d, n_embd_head, (int) grouped_ratio, num_groups_q, n_tokens,
                                       Q_4d->nb[1], Q_4d->nb[2], Q_4d->nb[3], 0);
    ggml_tensor * Q1    = ggml_cont_3d(ctx0, Q1_4d, n_embd_head, (int) grouped_ratio * num_groups_q, n_tokens);

    // Q2 = last head from each group → 8 heads (strided view)
    ggml_tensor * Q2_view = ggml_view_3d(ctx0, Q, n_embd_head, num_groups_q, n_tokens,
                                         Q->nb[1] * q_group_size,                                    // stride = 5
                                         Q->nb[2],
                                         n_embd_head * (int) grouped_ratio * ggml_element_size(Q));  // offset = 4
    ggml_tensor * Q2      = ggml_cont(ctx0, Q2_view);  // Make contiguous for concat

    // Cast K/V to F32 for concat operations (CUDA requirement)
    ggml_tensor * K_f32 = (K->type != GGML_TYPE_F32) ? ggml_cast(ctx0, K, GGML_TYPE_F32) : K;
    ggml_tensor * V_f32 = (V->type != GGML_TYPE_F32) ? ggml_cast(ctx0, V, GGML_TYPE_F32) : V;

    // Split K/V using grouped rearrangement (matching HuggingFace)
    // K/V: 16 heads → 8 groups of 2 → K1/V1 = first head per group, K2/V2 = second head per group
    const int kv_group_size = (int) k_ratio + 1;          // 2
    const int kv_num_groups = n_head_kv / kv_group_size;  // 8

    // K: [d, 16, N_kv] → [d, 2, 8, N_kv]
    ggml_tensor * K_4d = ggml_reshape_4d(ctx0, K_f32, n_embd_head, kv_group_size, kv_num_groups, n_kv);

    // K1 = first head from each group → 8 heads
    ggml_tensor * K1_4d = ggml_view_4d(ctx0, K_4d, n_embd_head, (int) k_ratio, kv_num_groups, n_kv, K_4d->nb[1],
                                       K_4d->nb[2], K_4d->nb[3], 0);
    // Make contiguous first, then reshape is no longer needed - just use cont_3d directly
    ggml_tensor * K1    = ggml_cont_3d(ctx0, K1_4d, n_embd_head, (int) k_ratio * kv_num_groups, n_kv);

    // K2 = second head from each group → 8 heads (strided view)
    ggml_tensor * K2_view = ggml_view_3d(ctx0, K_f32, n_embd_head, kv_num_groups, n_kv,
                                         K_f32->nb[1] * kv_group_size,                             // stride = 2 heads
                                         K_f32->nb[2],
                                         n_embd_head * (int) k_ratio * ggml_element_size(K_f32));  // offset = 1 head
    ggml_tensor * K2      = ggml_cont(ctx0, K2_view);                                              // Make contiguous

    // Same for V
    ggml_tensor * V_4d = ggml_reshape_4d(ctx0, V_f32, n_embd_head, kv_group_size, kv_num_groups, n_kv);

    ggml_tensor * V1_4d = ggml_view_4d(ctx0, V_4d, n_embd_head, (int) k_ratio, kv_num_groups, n_kv, V_4d->nb[1],
                                       V_4d->nb[2], V_4d->nb[3], 0);
    // Make contiguous first, then reshape is no longer needed - just use cont_3d directly
    ggml_tensor * V1    = ggml_cont_3d(ctx0, V1_4d, n_embd_head, (int) k_ratio * kv_num_groups, n_kv);

    ggml_tensor * V2_view = ggml_view_3d(ctx0, V_f32, n_embd_head, kv_num_groups, n_kv, V_f32->nb[1] * kv_group_size,
                                         V_f32->nb[2], n_embd_head * (int) k_ratio * ggml_element_size(V_f32));
    ggml_tensor * V2      = ggml_cont(ctx0, V2_view);  // Make contiguous

// Debugging output shapes
#if 1
    std::fprintf(stderr, "DEBUG LAYER %d:\n", il);
    std::fprintf(stderr, "Q: %ld %ld %ld\n", Q->ne[0], Q->ne[1], Q->ne[2]);
    std::fprintf(stderr, "Q1: %ld %ld %ld\n", Q1->ne[0], Q1->ne[1], Q1->ne[2]);
    std::fprintf(stderr, "Q2: %ld %ld %ld\n", Q2->ne[0], Q2->ne[1], Q2->ne[2]);
    std::fprintf(stderr, "K1: %ld %ld %ld\n", K1->ne[0], K1->ne[1], K1->ne[2]);
    std::fprintf(stderr, "K2: %ld %ld %ld\n", K2->ne[0], K2->ne[1], K2->ne[2]);
#endif

    // ========================================
    // PHASE 4: Fuse Q/K/V (HuggingFace approach)
    // ========================================
    const float kq_scale =
        hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    // q_f = concat([Q1, Q2]) → [d, 40, N]
    ggml_tensor * q_f = ggml_concat(ctx0, Q1, Q2, 1);

    // k_f = concat([repeat_interleave(K1, 4), K2]) → [d, 40, N_kv]
    // First repeat_interleave K1 from 8 heads to 32 heads
    // IMPORTANT: HuggingFace uses repeat_interleave which expands [A B C D] -> [A A A A B B B B C C C C D D D D]
    // NOT tiling which would be [A B C D A B C D A B C D A B C D]
    ggml_tensor * K1_rep4 = repeat_interleave_heads(ctx0, K1, (int) grouped_ratio);
    ggml_tensor * k_f     = ggml_concat(ctx0, K1_rep4, K2, 1);  // [d, 40, N_kv]  (K2 already contiguous)

    // v1_f = concat([repeat_interleave(V1, 4), V1]) → [d, 40, N_kv]
    ggml_tensor * V1_rep4 = repeat_interleave_heads(ctx0, V1, (int) grouped_ratio);
    ggml_tensor * v1_f    = ggml_concat(ctx0, V1_rep4, V1, 1);  // [d, 40, N_kv]  (V1 already contiguous from cont_3d)

    // v2_f = concat([repeat_interleave(V2, 4), V2]) → [d, 40, N_kv]  (V2 already contiguous)
    ggml_tensor * V2_rep4 = repeat_interleave_heads(ctx0, V2, (int) grouped_ratio);
    ggml_tensor * v2_f    = ggml_concat(ctx0, V2_rep4, V2, 1);  // [d, 40, N_kv]

#if 1
    std::fprintf(stderr, "PHASE 4 - Fused tensors:\n");
    std::fprintf(stderr, "q_f: %lld %lld %lld\n", (long long) q_f->ne[0], (long long) q_f->ne[1],
                 (long long) q_f->ne[2]);
    std::fprintf(stderr, "k_f: %lld %lld %lld\n", (long long) k_f->ne[0], (long long) k_f->ne[1],
                 (long long) k_f->ne[2]);
    std::fprintf(stderr, "v1_f: %lld %lld %lld\n", (long long) v1_f->ne[0], (long long) v1_f->ne[1],
                 (long long) v1_f->ne[2]);
    std::fprintf(stderr, "v2_f: %lld %lld %lld\n", (long long) v2_f->ne[0], (long long) v2_f->ne[1],
                 (long long) v2_f->ne[2]);
#endif

    // ========================================
    // PHASE 5: Two attention calls
    // ========================================
    // build_attn_mha flattens output to [d*heads, N] for o_proj
    // We need to reshape back to [d, N, heads] for merge/split operations

    // ========================================
    // PHASE 5: Two attention calls
    // ========================================
    // build_attn_mha flattens output to [d*heads, N]
    // The memory layout is [d, heads, N] (Token 0 [Head 0, Head 1...])
    // We MUST reshape to [d, heads, N] first to match layout

    ggml_tensor * attn_1_flat = build_attn_mha(q_f, k_f, v1_f, nullptr, kq_mask, nullptr, nullptr, kq_scale, il);
    cb(attn_1_flat, "attn_1_flat", il);

    ggml_tensor * attn_2_flat = build_attn_mha(q_f, k_f, v2_f, nullptr, kq_mask, nullptr, nullptr, kq_scale, il);
    cb(attn_2_flat, "attn_2_flat", il);

    // Correct Reshape: [d, heads, N]
    // nb[0]=4, nb[1]=d*4, nb[2]=d*heads*4 - matches attn_flat layout
    ggml_tensor * attn_1 = ggml_reshape_3d(ctx0, attn_1_flat, n_embd_head, n_head, n_tokens);
    ggml_tensor * attn_2 = ggml_reshape_3d(ctx0, attn_2_flat, n_embd_head, n_head, n_tokens);

#if 1
    std::fprintf(stderr, "PHASE 5 - Attention outputs (Corrected Shape):\n");
    std::fprintf(stderr, "attn_1: %lld %lld %lld\n", (long long) attn_1->ne[0], (long long) attn_1->ne[1],
                 (long long) attn_1->ne[2]);
    std::fprintf(stderr, "attn_2: %lld %lld %lld\n", (long long) attn_2->ne[0], (long long) attn_2->ne[1],
                 (long long) attn_2->ne[2]);
#endif

    // ========================================
    // PHASE 6: Merge and split results
    // ========================================
    // merged = concat([attn_1, attn_2], dim=0)
    // concat along dim 0 (d) -> [2d, heads, N]
    ggml_tensor * merged_attn = ggml_concat(ctx0, attn_1, attn_2, 0);

    // attn_o = merged[:, :32, :] -> [2d, 32, N]
    // View stride nb[1] (heads) matches merged_attn
    ggml_tensor * attn_o = ggml_view_3d(ctx0, merged_attn,
                                        n_embd_head * 2,                     // ne[0] = 2d
                                        (int) grouped_ratio * num_groups_q,  // ne[1] = 32 heads
                                        n_tokens,                            // ne[2] = N
                                        merged_attn->nb[1], merged_attn->nb[2],
                                        0);                                  // offset 0

    // attn_n_group = merged[:, 32:, :] -> [2d, 8, N]
    ggml_tensor * attn_n_group = ggml_view_3d(ctx0, merged_attn,
                                              n_embd_head * 2,  // ne[0] = 2d
                                              num_groups_q,     // ne[1] = 8 heads
                                              n_tokens,         // ne[2] = N
                                              merged_attn->nb[1], merged_attn->nb[2],
                                              n_embd_head * 2 * ((int) grouped_ratio * num_groups_q) *
                                                  ggml_element_size(merged_attn));  // offset = 32 * stride_head

    // attn_n = repeat(attn_n_group, 4) -> [2d, 32, N]
    // Repeat along dim 1 (heads)
    ggml_tensor * attn_n_target =
        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_head * 2, (int) grouped_ratio * num_groups_q, n_tokens);
    ggml_tensor * attn_n = ggml_repeat(ctx0, attn_n_group, attn_n_target);

#if 1
    std::fprintf(stderr, "PHASE 6 - Split:\n");
    std::fprintf(stderr, "merged_attn: %lld %lld %lld\n", (long long) merged_attn->ne[0],
                 (long long) merged_attn->ne[1], (long long) merged_attn->ne[2]);
    std::fprintf(stderr, "attn_o: %lld %lld %lld\n", (long long) attn_o->ne[0], (long long) attn_o->ne[1],
                 (long long) attn_o->ne[2]);
    std::fprintf(stderr, "attn_n: %lld %lld %lld\n", (long long) attn_n->ne[0], (long long) attn_n->ne[1],
                 (long long) attn_n->ne[2]);
#endif

    // PHASE 7: Differential
    // ========================================
    // lambda_full = lambda_diff + lambda_init
    ggml_tensor * attn_n_scaled_diff = ggml_mul(ctx0, attn_n, lambda_diff);
    ggml_tensor * attn_n_scaled_init = ggml_scale(ctx0, attn_n, lambda_init);
    ggml_tensor * attn_n_scaled      = ggml_add(ctx0, attn_n_scaled_diff, attn_n_scaled_init);

    ggml_tensor * attn_diff = ggml_sub(ctx0, attn_o, attn_n_scaled);
    // attn_diff: [2d, 32, N]

    // ========================================
    // PHASE 8: SubLayerNorm - BEFORE flattening!
    // ========================================
    // HuggingFace: self.subln(attn_output) where attn_output is [bsz, q_len, 32, 2d]
    // The subln weight is [2d] = [256], normalizing across the head_dim*2 dimension
    // ggml_rms_norm normalizes along ne[0] which is 2d (256) - CORRECT!
    ggml_tensor * attn_norm = ggml_rms_norm(ctx0, attn_diff, hparams.f_norm_rms_eps);
    // attn_sub_norm has shape [2d] = [256], broadcast across heads and tokens
    attn_norm               = ggml_mul(ctx0, attn_norm, attn_sub_norm);

    // Final scale: (1 - lambda_init)
    ggml_tensor * attn_scaled = ggml_scale(ctx0, attn_norm, 1.0f - lambda_init);

    // NOW flatten for output projection: [2d, 32, N] -> [2d*32, N] = [8192, N]
    ggml_tensor * attn_flat = ggml_cont_2d(ctx0, attn_scaled, n_embd_head * 2 * n_head_q1, n_tokens);

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
