# Guide: Adding New Model Architectures

*Author: [pwilkin](https://github.com/pwilkin)*

## Intro

With my adventures with Qwen3 Next ([#16095](https://github.com/ggml-org/llama.cpp/pull/16095)), which some of you might have seen, I think I learned a lot about the Llama.cpp / GGML internal architecture / quirks, so I decided to write a guide for someone who might want to try, like me, to dust off their C++ and attempt to implement a new model architecture.[1]

## Prerequisites

While nowadays a lot of people vibe code stuff with the help of LLMs and it's possible to do a lot of things outside the area of your expertise, there are nevertheless a couple of skills that are must haves for coding a model architecture.[1]

- Basic knowledge of how an LLM works – you need to know what tensors and layers are, how data gets encoded and decoded in an LLM.[1]
- Linear algebra – this is non-negotiable. If you've never done matrix multiplication in your life, you're going to struggle mightily. The entry level isn't very high, just basic matrix operations will do, but you have to be comfortable with it.[1]
- Pointer arithmetic and basic C/C++ – GGML is a low-level library. You will have to manage the memory layout of tensors and properly write values to the tensors yourself if you ever write an operation (more on that later).[1]
- Patience – lots and lots of it. If you're doing a harder model, you might hit a wall where you have no idea why your code is not working correctly. You'll be comparing tensor dumps and wondering where that 7.5251 vs 7.4621 sum divergence comes from. The less experienced you are, the more you'll have to do it.[1]

## Where to start

Obviously, unless you're willing to add a new model architecture from scratch in GGML (feel free, but it's a pretty challenging task!), you'll be working off some reference implementation.  Now, for various reasons, the best reference implementation choice is the Transformers implementation, if there is one.[1]

First, there is tooling that lets you easily work with a Transformers implementation to compare your adaptation; second, there is know‑how on how to do various things that other frameworks might lack.  If you have to pick another reference implementation than Transformers, the difficulty level rises significantly.[1]

Also, be sure you pick a stable reference implementation.  Sometimes, even the model developers themselves leave bugs in their implementations that they will fix later on. Make sure the reference implementation is actually stable and producing good results before you attempt the conversion.[1]

## The key elements

### `convert_hf_to_gguf_update.py` (new tokenizer)

If your model features a new tokenizer – one that hasn't been used before – you have to update `convert_hf_to_gguf_update.py` to fetch the tokenizer from a reference model on Hugging Face, then run the script.  If your tokenizer is already supported, you will skip this step.  Do not add a new tokenizer if it's the same tokenizer as one already used.[1]

### `convert_hf_to_gguf.py`

This is where any conversion starts: the conversion script, one huge file where all the converters reside.[1]

However, this isn't the only file you need to look at before the conversion.  Probably, the ones you should start with – before you even write a line of code in `convert_hf_to_gguf.py` – are `constants.py` and `tensor_mapping.py` inside the `gguf-py` package.  They contain the currently supported tensor types in GGML as well as the parameter names.[1]

You should look at the `model.safetensors.index.json` and `config.json` files from the original model (assuming you're converting from safetensors) to get the list of layers and the list of parameters, respectively.  The first thing you should do is make a checklist:[1]

- Do all the tensors in my model have their representations in GGML? If not, do not worry – you can add new ones. But before you do that, make sure that there isn't a representation named in a non‑obvious way – at least Ctrl+F `tensor_mappings.py` to check if a tensor with the same name scheme was previously handled in some other model.[1]
- Which parameters will you need? Remember that, unless you're feeling very ambitious, you're aiming for forward‑pass compatibility – you will not usually be providing full backward‑pass capability for model training. You can take two approaches – either look at the `modeling_*.py` class from the Transformers implementation to check which parameters you will need, or you can just add parameters on the fly when needed. A lot of the parameters will be handled by default implementations, so keep that in mind – you do not have to manually handle all of the parameters.[1]
- What, if any, preparatory work on the tensors will have to be done? One very important thing is how GGML handles expert tensors – the standard way is for the expert tensors to be merged into one big expert tensor. If you're unsure how, look at how already implemented MoE architectures handle this. But also look for places where you can optimize by preprocessing the tensors. If your tensor is always subject to preparatory operations before usage (for example, the Transformers code always does `exp(your_tensor)` or `1 + your_tensor` or similar), then you should do that in the preparation code to both simplify the graph building and optimize (since you will only do the calculation once). The reason why this is sometimes not done in the reference code is because of the backward pass, but here you're aiming for forward pass capability most of the time.[1]

Once you've looked at the reference code, tensors and parameters, it's time to write the code.  You will be modifying the following places:[1]

- `constants.py` – the model architecture constant, its codename and the list of tensors that it uses.[1]
- `tensor_mapping.py` – if the names of the tensors in your model follow a non‑standard pattern, i.e. do not match any of the patterns already listed, add the mappings in that file. Unless some tensors are completely specific to your model, add the mappings in the general case.[1]
- `llama-arch.h` and `llama-arch.cpp` – the architecture and name, to mirror the one in `constants.py`.[1]
- `convert_hf_to_gguf.py` – the conversion class itself.[1]

When writing the conversion class, make sure to inherit from the closest class that mirrors your architecture (`TextModel` at the minimum).  Usually, you can just copy‑paste the additional parameters, expert packing and other tensor conversion code from existing examples, but the methods you want to pay attention to are:[1]

- `set_gguf_parameters` – where you convert all the non‑standard hyperparameters.[1]
- `prepare_tensors` – where you create any new tensors that might be needed.[1]
- `modify_tensors` – where you perform any processing on existing tensors (merge, omit, transform etc.). One important thing to note is that GGML tensors that have a weight and bias part have to follow the `"X.weight"` and `"X.bias"` convention, so if the convention is different (for example `"X_weight"` and `"X_bias"` or `"X"` and `"X_bias"`), you will have to modify the names to match the convention.[1]

Once you're done, you should attempt to convert the model.  Pay attention to any error messages suggesting that there are tensors you're not handling.  If there are tensors you do not want to handle (for example because they correspond to functionality you are not converting, i.e. MTP), you explicitly have to ignore them in `modify_tensors`.[1]

### `llama-model.cpp`

Now comes the most time‑consuming part – the graph implementation.  You will most certainly want to look inside `examples/model-conversion` – there are scripts there for running a reference implementation together with your implementation to get the logits and compare.[1]

However, logits might not be enough – if you struggle with conversion, you might want to analyze tensor dumps.  The `run-org-model` script has been modified to provide GGML‑style tensor dumps, but you might want to monkey‑patch some more functions specific to your model (a commented example is in the script file).[1]

The `llama-model.cpp` file itself houses just the code for loading the tensors and parameters.  For the specific graph builder for your model, you should create a new file inside `src/models` (remember to add it to `src/CMakeFiles.txt` afterwards) and add the declaration of your class to `src/models/models.h`.[1]

## What is a graph builder?

Before even starting, a key note on how inference works in Llama.cpp.  When you look at Transformers code, you can basically figure out how the processing goes – the model's main `forward` method is called, which chains the `forward` calls of the submodules for the separate layers and components within. But that is **not** how inference works in GGML / Llama.cpp.[1]

In Llama.cpp, due to its support of multiple different backends, which are supposed to be transparent to the end user, a different approach is used.  What the model architecture constructs is not the inference itself but the **inference graph**.[1]

This means a couple of things.[1]

- Whenever you code any instructions that operate on tensors, what happens when those instructions are called is that the respective operation gets added to the graph, not executed.  That means that, apart from the tensor type, dimensions and strides, you cannot obtain any information from the tensor at the stage of building the graph itself – notably, you cannot extract any data.[1]
- The graph has to be static. Its shape can depend on hyperparameters of the model or metaparameters such as layer number, but it cannot depend on the values of the processed tensors at the time of processing. Any such dependencies have to be encoded into the processing itself – into entities known as **operations**.[1]
- Graph building is done lazily – if a node is deemed unreachable, it will not be computed. You have to mark which nodes are actually important for your processing – which is done using the function `ggml_build_forward_expand`. This function marks a graph node as essential in the computation, which means the node – and all nodes along the path to that node – will be computed. This still does not perform the computation itself and there is no method in the graph builder to force computation, because the graph is essentially a template and not a singular instance of an inference.[1]
- You cannot perform loops. Even if you can make the loop logic independent of specific tensor values, looping over e.g. the number of tokens, when a batch can have, say, 512 tokens, will generate a subgraph with `512 * (number of nodes inside the loop)` subnodes. This is not feasible and will exhaust memory instantly. If you need a loop, you have to do it inside a custom operation.[1]

## GGML quirks and specifics

There are two major differences between GGML and Transformers that you absolutely have to be aware of:[1]

- GGML tensor indexing is **little‑endian**. This is opposite to Transformers, which is **big‑endian**. Both are “row‑major” (meaning that a tensor of dimensions (5, 4) is going to be “five rows of four elements”). But a tensor of dimensions (5, 4) in Transformers will actually have dimensions (4, 5) in GGML. Basically, whenever you have a Transformers tensor shape, you reverse it to get a GGML shape. Also, GGML tensors are strictly limited to 4 dimensions (actually, they are technically all 4‑dimensional), so if you need to represent a 5‑dimensional (or more) Transformers tensor, you have to manually pack it. This also means that the default 4‑dimensional tensor ordering in transformer models, which is typically (number of sequences, sequence size in tokens, dimension 1, dimension 2) will be reversed in GGML: (dimension 2, dimension 1, sequence size in tokens a.k.a. `n_seq_tokens`, number of sequences a.k.a. `n_seq`).[1]
- Matrix multiplication in GGML is also different. The `matmul` operation in GGML: `matmul(A, B)` corresponds to `transpose(B) @ A` in Transformers or, in other terms, `transpose(matmul(A, B)) = transpose(A) @ B`. Most other operations mirror their Transformers counterparts, but matrix multiplication is the one which stands out.[1]

## Model preparation

Before you can do any graph building, the graph needs to have a model to operate on.  You converted the tensors, now you need to load them into a layered model structure.[1]

First, load the hyperparameters into the GGML `hparam` structures, which is done in `llama-model.cpp` in the `load_hparams()` method.  Then, load the tensors themselves – this is done in the huge switch statement within the `load_tensors()` method.  This is where you actually load the tensors you converted and specify their dimensions using the hyperparameters.[1]

All of the tensors have to actually get loaded, or the loader will throw an error.  This will be the first real test of the conversion – whether you can reasonably load the tensors you converted with proper dimensions that make semantic sense within the model parameters.[1]

Finally, if your model has a novel size, make sure to update the model sizes enum.[1]

## Graph building

Now you get to the non‑trivial part: building the graph.  Note that GGML already has functions for some typical things that happen during an LLM's forward inference – such as RoPE, expert routing, standard attention, KV cache handling, input position embeddings.  Unless you're absolutely sure that the architecture you're building for does things completely differently, assume that you will be using those functions.[1]

Do not try to mirror the Transformers code step‑by‑step – instead, understand what the building blocks are.  Most typical tensor operations have their counterparts in GGML.  Here are a couple of typical ones:[1]

- Matrix multiplication (`@`) → `ggml_mul_mat` (but see the note above), also `build_mm_lora` specifically for weight projections which might use LoRAs.[1]
- Standard element‑wise multiplication → `ggml_mul` (this has broadcasting, but only in one dimension).[1]
- Repeat → `ggml_repeat_4d` (broadcasts the tensor to fill the given shape).[1]
- Pad → `ggml_pad` (semantics are different: `ggml_pad` does post‑padding and takes the pad value in each dimension, whereas standard Transformers `pad` only does padding on the semantic dimensions, and takes pre‑ and post‑pad arguments). For pre‑ and post‑padding, see `ggml_pad_ext`.[1]
- Reshape → `ggml_reshape_(1|2|3|4)d`. [1]  
- Contiguous → `ggml_cont` (or `ggml_cont_(1|2|3|4)d`, which is reshape + cont). [1]  
- Transpose → `ggml_permute` or `ggml_transpose` – the former takes a permutation tuple or the indices where the respective dimensions will land (e.g. `ggml_permute(ctx, t, 3, 0, 1, 2)` means a permutation of 0 → 3, 1 → 0, 2 → 1, 3 → 2); the latter just exchanges the semantic dimensions (the first two).[1]
- Scalar multiplication and addition → `ggml_scale`, `ggml_scale_bias`.[1]
- `exp` → `ggml_exp`.[1]
- New zero tensor → `ggml_new_tensor_(1|2|3|4)d` (tensor is zeroed by default). [1]  
- New ones tensor → `ggml_exp(ggml_new_tensor)` (since \(e^0 = 1\)).[1]
- Tensor slices → `ggml_view_(1|2|3|4)d` – you have to provide strides (number of bytes by which you jump to get the next element on that dimension) and offset. Remember to use `ggml_element_size` and `ggml_nelements` instead of fixed data types. Unless you're doing really weird views, passing the original tensor strides (held in `t->nb` array) will be enough. If you're looking for dimensions (the equivalent of Transformers' `.shape`), they're in `t->ne` (so the full shape of the tensor is `(t->ne[0], t->ne[1], t->ne[2], t->ne[3])`). [1]

## New operations

If you find an operation that isn't implemented, you can try to work around it with equivalent transformations or add the operation.  Adding an operation is a separate topic in itself, but the standard is to provide a CPU backend implementation as the reference one and keep any further backends for separate PRs.[1]

Basically, the operations are the code that actually gets executed once the graph is computed and inference is done.  Once you're coding operations, you're out of the world of abstractions and in the world of pointer arithmetic, memory allocations, manually traversing tensor dimensions and the like.[1]

There are abstractions coded in the library; for example, there are C++ templates if your operation is a true unary operation (operates independently on tensor elements element‑by‑element, without extra parameters, such as NEG or EXP) or binary operation (operates element‑wise on element‑pairs from two tensors).  To add an operation, you have to:[1]

- Add the method (conventionally called `ggml_...`) which actually queues the operation by preparing the result tensor and passing the arguments – either as source tensors in the `t->src` array or in the dedicated parameters array as `t->op_params`.[1]
- Add the operation itself to `ggml.h` and `ggml.c`, and do not forget to increase the static count assertions; if it's a true unary op, don't add it to the ops enum, but to the unary ops enum.[1]
- Add the implementation call to the switch in `ggml-cpu.c` and implement the operation itself (likely in `ops.cpp` or `unary-ops.cpp`).[1]
- Add some basic tests to `test-backend-ops`.[1]

## Which RoPE?

This is a short but potentially annoying topic.  There are basically two types of RoPE used: normal and so‑called “NeoX”.  However, determining which is which is counterintuitive.[1]

When you have a Transformers implementation of `apply_rotary_pos_emb`, it will usually look like this (simplified):[1]

```python
original_dtype = q.dtype
cos = cos.unsqueeze(unsqueeze_dim)
sin = sin.unsqueeze(unsqueeze_dim)

# Interleave them instead of usual shape
cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)

q_embed = (q.float() * cos) + (rotate_half(q).float() * sin)
k_embed = (k.float() * cos) + (rotate_half(k).float() * sin)

return q_embed.to(original_dtype), k_embed.to(original_dtype)
```

The critical part is in these two lines:[1]

```python
cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
```

If those lines **are** present, it's **normal** RoPE.  If they **aren't** present, it's **NeoX** RoPE.  Mark the appropriate case in the switch at the end of `llama-model.cpp`.[1]

## Debugging

That's it.  Now comes the tedious part of making sure your implementation works well.[1]

At a minimum, the conversion should pass the test in `examples/model-conversion` from the compare logits task.  You should also verify that it produces coherent long output and that it can coherently read long inputs.[1]

If your conversion matches the reference perfectly on short prompt processing but diverges on generation, it's usually a sign of either:[1]

- Incorrect state management (can happen especially in mamba / hybrid models).[1]
- Bad RoPE (see above; also make sure the hyperparams for RoPE are correct, especially if using YaRN).[1]

From experience, the biggest culprits in divergence are incorrect tensor shapes and/or transpositions, which quickly lead to diverging outputs.  A good heuristic when looking at tensor dumps is to check whether the top‑left and bottom‑right corners match – e.g. elements `(2, 0), (1, 0), (0, 0), (0, 1), (0, 2)` and their bottom‑right counterparts – which can quickly reveal obvious transposition or tensor‑ordering problems.[1]

For easily gathering tensor dumps on single token processing, use `llama-eval-callback`.  If you need to debug longer sequences, you might need to copy the callback code from `llama-eval-callback` to `llama-cli` and enable the callback by passing the appropriate params (see `eval-callback.cpp` for details).[1]

## Prompt template

If your model uses a non‑typical thinking marker, non‑typical tool‑calling markers or both, you have to add chat template support in `chat.cpp` and `llama-chat.cpp`.  This means, basically:[1]

- Adding a means of identifying the specific chat format based on unique fragments of the Jinja template.[1]
- Adding a grammar for whenever you want Llama.cpp to force certain outputs (such as when generating tool calls).[1]
- Adding a parser to parse whatever the model outputs.[1]
- Making sure the Jinja template is fully supported (Llama.cpp uses Minja, which does not support all Jinja features, so you might have to replace some constructs in the template with equivalent ones).[1]

Unless your model is especially nasty, this basically means copying what has already been done with a similar model.  Make sure to add the necessary tests to `test-chat.cpp` to verify if your chat parser works correctly.[1]

## And that's it!

If you made it this far, congratulations.  Your basic model architecture implementation is done.  Now you can focus on adding those optimized CUDA kernels, Vulkan shaders and all the other things that aren't critical for correctness, but can make the model work faster.[1]

[1](https://github.com/ggml-org/llama.cpp/discussions/16770)