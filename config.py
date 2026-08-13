"""Every adjustable setting for tokenization, modeling, training, and generation."""


CORPUS_PATH = "data/raw/notes.txt"  # Raw text used to train the tokenizer and model.
AUGMENTED_CORPUS_PATH = "data/generated/augmented_notes.txt"  # Original notes plus generated paraphrases used when available.
PARAPHRASE_PROGRESS_PATH = "data/generated/paraphrases.jsonl"  # Completed API results used to resume an interrupted run.
TOKENIZER_PATH = "tokenizer.json"  # Saved BPE merges and token vocabulary.
CHECKPOINT_PATH = "model.pt"  # Saved model parameters loaded during generation.

PARAPHRASE_MODEL = "gpt-5-mini"  # Low-cost OpenAI model used for the narrow paraphrasing task.
PARAPHRASE_VARIANTS = 2  # New rewrites per source chunk; two plus the original makes about 3x text.
PARAPHRASE_CHUNK_CHARACTERS = 12_000  # Approximate chunk size, split only between top-level notes.
PARAPHRASE_MAX_OUTPUT_TOKENS = 6_000  # Safety ceiling for the response to one source chunk.

END_OF_TEXT = 256  # Token ID reserved for the end of a document.
VOCAB_SIZE = 1024  # Total number of byte, special, and learned BPE tokens.

CONTEXT_LENGTH = 256  # Maximum number of preceding tokens visible to the model.
EMBEDDING_SIZE = 256  # Number of features used to represent each token.
NUMBER_OF_HEADS = 4  # Independent attention patterns computed in each block.
NUMBER_OF_BLOCKS = 4  # Transformer blocks applied one after another.
FEED_FORWARD_MULTIPLIER = 4  # Expansion factor inside each feed-forward network.
DROPOUT = 0.1  # Fraction of activations randomly hidden during training only.
WEIGHT_INIT_STD = 0.02  # Standard deviation for initial random model weights.

BATCH_SIZE = 32  # Independent context windows processed in one optimizer step.
TRAINING_STEPS = 3000  # Optimizer steps targeting roughly 30 minutes on this M2 Air.
LEARNING_RATE = 3e-4  # Peak size of each AdamW parameter update.
MINIMUM_LEARNING_RATE = 3e-5  # Smallest update size reached near the end of training.
WARMUP_STEPS = 100  # Steps spent gradually increasing to the peak learning rate.
LEARNING_RATE_DECAY_STEPS = 3000  # Step at which cosine decay reaches its minimum.
WEIGHT_DECAY = 0.1  # Strength of AdamW's pressure toward smaller weights.
TRAINING_FRACTION = 0.9  # Fraction of tokens used for training instead of validation.
EVALUATION_INTERVAL = 100  # Training steps between loss reports.
EVALUATION_BATCHES = 20  # Random batches averaged for each reported loss.
TRAINING_SAMPLE_TOKENS = 50  # New tokens printed after each training evaluation.
RANDOM_SEED = 42  # Seed making initialization and batch sampling reproducible.

DEFAULT_PROMPT = "- i think"  # Prompt used for training samples and default generation.
TOKENS_TO_GENERATE = 200  # Maximum new tokens sampled during default generation.
TEMPERATURE = 0.7  # Sampling randomness; lower values favor likely tokens.
