import numpy as np
import torch
from .stream import stream_sentences

def create_skipgram_generator(file_path, word_to_ix, window_size=5):
    """
    A generator that yields (target, context) word index pairs for skip-gram training.
    This function streams the data and does not store all pairs in memory.
    """
    for sentence in stream_sentences(file_path):
        for i, target_word in enumerate(sentence):
            if target_word not in word_to_ix:
                continue
            target_ix = word_to_ix[target_word]
            
            # Dynamically set the window size, similar to gensim
            reduced_window = np.random.randint(1, window_size + 1)
            
            start = max(0, i - reduced_window)
            end = min(len(sentence), i + reduced_window + 1)
            
            for j in range(start, end):
                if i == j:
                    continue
                context_word = sentence[j]
                if context_word not in word_to_ix:
                    continue
                context_ix = word_to_ix[context_word]
                yield (target_ix, context_ix)

def create_skipgram_batch_generator(file_path, word_to_ix, window_size, batch_size):
    """
    A generator that yields BATCHES of (target, context) word index pairs.
    """
    batch_targets = []
    batch_contexts = []
    
    # Use the single-pair generator as a source
    for target_ix, context_ix in create_skipgram_generator(file_path, word_to_ix, window_size):
        batch_targets.append(target_ix)
        batch_contexts.append(context_ix)
        
        # When the batch is full, yield it and reset
        if len(batch_targets) >= batch_size:
            # Convert to tensors before yielding
            target_tensor = torch.tensor(batch_targets, dtype=torch.long)
            context_tensor = torch.tensor(batch_contexts, dtype=torch.long)
            yield target_tensor, context_tensor
            
            # Reset for the next batch
            batch_targets = []
            batch_contexts = []
            
    # Don't forget the last, possibly incomplete, batch
    if len(batch_targets) > 0:
        target_tensor = torch.tensor(batch_targets, dtype=torch.long)
        context_tensor = torch.tensor(batch_contexts, dtype=torch.long)
        yield target_tensor, context_tensor
    