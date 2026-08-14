# TadpoleLM

## Description

This is a tiny, proof-of-concept LLM I created based on my personal notes (not included here), similar in scale and nature to Karpathy's NanoGPT. Hopefully, we can extend this to something enabling more fun interpretability projects. 

- the corpus being used is my personal notes. i wrote it in a very conversational style, which i'm intentionally mimicking here
   - lots of "quotes", other fun bad grammar stuff
   - there's also bullets and indents. these all get recreated in the model!
- the inference prompt is "- i think "
- since it's a super small corpus, i used an LLM to paraphrase and augment it (script but not results are available)

Check out the training run artifacts; you can watch the model learn (and overfit) in real time! The configs used to generate each model are also available. I outlined the training and inference workflows below. 

This code is AI-generated; this README isn't. 

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

## Sample Output (500 tokens, temperature=0.7)


- i think the eout of that i don’t want to be a lot of my strictly be the outcomesthetically content
- just not getting about how most people and on untiles to meaning too is an infascinating
    - but not gruances might be a look it’s annoying it’s an in-pretty relativity
    - the questioning direction of the led meaning that it output and then you should be such i have to learn the untist’s an evaluates you could come ast up needing it
    - i notice force becomes it
    - just makes even if you’re all about a dumb inflearing and feels domaint and processential to reading it
    - “what relyabout way” and then ether novel way
    - the awaste to becide and often “ware about this” 
    - forwaste finit’s become mave anularity 
- math choices that really active write can be some experience
    - you’re doing technically the decision, but that’s cument
    - i chany of the comeralental of gentered to talking is funny, but obviously
    - you can resolving want to company funny 
    - like, that it’s a lot of the problem is do talkead, like a other was storgirds for AI, derenageepends nsimple often up without responses? 
    - which makes i ambigin much of the way i feel like inno much as my own outcomes from the nerkind of about a hadent of clase of jargon remon, and genuinely, whimse force of a largony p a lue twarian kid, and no meta flongersame exect or something i like show, ie their frich levels of idea jud
