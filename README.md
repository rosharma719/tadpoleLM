# TadpoleLM

## Description

This is a series of tiny, proof-of-concept LLMs I created based on my personal notes (not included here), similar in scale and nature to Karpathy's NanoGPT, and then extended with more scale. 

- the corpus being used is my personal notes. i wrote it in a very conversational style, which i'm intentionally mimicking here
   - lots of "quotes", other fun bad grammar stuff
   - there's also bullets and indents. these all get recreated in the model!
- the inference prompt is "- i think "
- since it's a super small corpus, i used an LLM to paraphrase and augment it (script but not results are available)

We started with an 11M model that I pretrained on my notes. With the second model (tadpole-english-30m) and subsequent model (tadpole-notes-30m), we extended to a 30M parameter model that we first pretrained on HF's FineWeb-Edu. The tadpole-notes-30m was pretrained on teh English dataset and then fine-tuned on my notes. The 30M models were trained on vast.ai GPUs.

Check out the training run artifacts for the original model (not on HF); for the 11M model, you can watch it learn (and overfit) in real time! The configs used to generate each model are also available. I outlined the training and inference workflows below. 

Models available at https://huggingface.co/rosharma719. Check out the artifacts folder to read some of their output!


## Training Workflow
1. Load tokenizer.json and replay learned BPE merges. This returns one long 1D tensor (of integer token IDs, corresponding to learned tokens)
2. Split training and validation data (~90/10)
3. Build the model; randomly initialize 3.5m params, move to Apple GPU. The model contains token embeddings, position embeddings, transformer blocks, the final layer norm, and the language-model head/unembedding matrix.
4. Build the optimizer
5. Evaluate at Step 0 (random) and construct a training batch
6. Choose 32 random starting positions and call the model. Batch size = 32, sequence length = 256, embedding size = 256.
7. Run token/position embeddings and transformer blocks. This includes attention, followed by an output matrix and layer normalization, then a FFN, then layer norm. 
8. After 4 blocks and final layer norm, we map each 256-dimensional token vector into 1024 token scores. 
9. Calculate loss, which takes logits and not probabilities because it's cross-entropy. Cross entropy: converts logits into log probabilities, selects probability assigned to correct target, calculates -log(correct prob), and averages across all 32*256=8192 predictions. 
10. Backpropagation (AdamW)

## Inference Workflow
1. Load prompt, tokenizer, model
2. Start with batch size 1, predict 1 new token. This uses logits[:, -1, :], which is the full token vector of the final sequence index in all batches (just the one batch). 
3. Continue generating; next token logits are divided by temperature. Temperature < 1 means the difference between logits grows, so bigger logits become more likely. We also have to convert logits to probabilities using softmax. Append to the sequence and repeat generation. 

