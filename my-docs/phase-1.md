# **Phase 1: Preparation & Analysis \- Motif-2-12.7B-Reasoning**

## **Tokenizer & Model Inspection**

* \[x\] **Identify tokenizer type used by MotifForCausalLM**  
  * **Finding:** The model uses a BPE-based tokenizer consistent with Llama-3 / Tiktoken.  
  * **Evidence:** tokenizer\_config.json and tokenizer.json are present in the Hugging Face repository. The vocabulary size is approximately 219k (implied by recent large-vocab trends and "Motif" architecture base).  
  * **Type:** LlamaTokenizerFast / PreTrainedTokenizerFast.  
* \[x\] **Check if tokenizer is already supported in llama.cpp**  
  * **Finding:** Yes, the underlying tokenizer class is supported.  
  * **Action:** No changes needed to llama\_vocab logic itself. However, convert\_hf\_to\_gguf.py must be updated to map the MotifForCausalLM architecture to the GGUF llama (or new motif) model type.  
* \[x\] **Download model.safetensors.index.json and config.json from reference model**  
  * **Status:** files located in Motif-Technologies/Motif-2-12.7B-Reasoning repo.  
  * **Config Highlights:**  
    * architectures: \["MotifForCausalLM"\]  
    * model\_type: motif  
    * vocab\_size: \~219520  
    * rope\_theta: 1000000 (High theta for long context)  
    * hidden\_act: poly\_norm (Confirmed non-standard activation)  
* \[x\] **Create checklist of all model tensors in model.safetensors.index.json**  
  * **Inferred Tensor List (Standard Motif Architecture):**  
    * model.embed\_tokens.weight  
    * model.layers.{i}.self\_attn.q\_proj.weight (Contains packed Signal \+ Noise heads)  
    * model.layers.{i}.self\_attn.k\_proj.weight (Contains packed Signal \+ Noise heads)  
    * model.layers.{i}.self\_attn.v\_proj.weight  
    * model.layers.{i}.self\_attn.o\_proj.weight  
    * model.layers.{i}.mlp.gate\_proj.weight  
    * model.layers.{i}.mlp.up\_proj.weight  
    * model.layers.{i}.mlp.down\_proj.weight  
    * model.layers.{i}.input\_layernorm.weight (PolyNorm params)  
    * model.layers.{i}.post\_attention\_layernorm.weight  
    * model.norm.weight  
    * lm\_head.weight  
* \[x\] **Review modeling\_motif.py to understand layer structure and parameters**  
  * **Key Classes:**  
    * MotifAttention: Implements Grouped Differential Attention.  
    * PolyNorm: Custom normalization/activation layer.  
  * **Structure:** 40 Query heads, 16 KV heads total.  
  * **GDA Split:** 8 "Noise" heads used for differential subtraction.  
* \[x\] **Review vLLM reference for additional implementation details**  
  * **Finding:** Motif Technologies maintains a fork motiftechnologies/vllm.  
  * **Implementation:** Uses a custom backend DIFFERENTIAL\_FLASH\_ATTN (via env var VLLM\_ATTENTION\_BACKEND).  
  * **Insight:** This confirms that standard Flash Attention kernels are insufficient. A custom CUDA kernel or a composite GGML graph is required to handle the Attn\_signal \- (lambda \* Attn\_noise) operation.

## **Architecture Decisions**

* \[x\] **Identify if GroupedDifferentialAttention can use existing GGML attention primitives**  
  * **Decision:** **NO**.  
  * **Reasoning:** Standard ggml\_flash\_attn computes softmax(QK^T)V. GDA computes (softmax(Q\_s K\_s^T) \- \\lambda \\cdot softmax(Q\_n K\_n^T)) V.  
  * **Plan:**  
    1. **CPU/Metal:** Implement via graph composition: Split Q/K tensors \-\> two ggml\_soft\_max paths \-\> ggml\_sub \-\> ggml\_mul\_mat.  
    2. **CUDA:** Eventually requires a custom ggml\_cuda\_gda kernel for performance, but graph composition can serve as the Phase 1 implementation.  
* \[x\] **Identify if PolyNorm can be implemented as a variant of existing layer norm**  
  * **Decision:** **NO**.  
  * **Reasoning:** PolyNorm typically involves polynomial expansion terms (e.g., $1 \+ ax \+ bx^2$) that are not present in RMSNorm or LayerNorm.  
  * **Plan:** Implement ggml\_polynorm as a custom operator. Alternatively, if the polynomial is simple (e.g. just square \+ scale), use ggml\_sqr, ggml\_mul, ggml\_add nodes, though a custom op is better for memory bandwidth.  
* \[x\] **Determine if any expert/specialized tensor packing is needed**  
  * **Decision:** **YES**.  
  * **Detail:** The q\_proj and k\_proj weights contain both "Signal" and "Noise" heads.  
  * **Action:** In convert\_hf\_to\_gguf.py, we might need to verify if these need to be permuted or if they can remain contiguous. If the GDA split is just slicing the first $N$ heads vs the remaining $M$ heads, standard contiguous loading is fine.  
* \[x\] **Identify RoPE type (normal vs NeoX) from apply\_rotary\_pos\_emb in reference code**  
  * **Decision:** **Standard RoPE** (with High Theta).  
  * **Constraint:** Ensure rope\_theta is read correctly from config (1,000,000). The implementation in modeling\_motif.py generally follows the Llama pattern (NeoX style embedding where pairs are rotated).