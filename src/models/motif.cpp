#include "models.h"
llm_build_motif::llm_build_motif(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    // TODO: Implement Motif graph builder
    GGML_UNUSED(model);
    
    // Minimal stub that just returns embeddings without actual processing
    // This will allow the model to load but won't produce correct outputs
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "result_embd", -1);
    cur = build_lora_mm(model.output, cur);
    res->t_logits = cur;
}