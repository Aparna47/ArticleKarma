import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


# --- Model Definition (same as before) ---
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, word_to_ix, ix_to_word):
        super(Word2Vec, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Linear(embedding_dim, vocab_size)

        # Initialize the cache for our "singleton" normalized matrix
        self._normalized_embeddings_cache = None

    def forward(self, target_word_ix):
        # Invalidate the cache during a forward pass in training
        if self.training:
            self._normalized_embeddings_cache = None
        embedded = self.in_embeddings(target_word_ix)
        scores = self.out_embeddings(embedded)
        return scores

    # --- Overriding train() and eval() to handle caching ---
    def train(self, mode=True):
        """
        Overrides the default train method to clear the cache when switching to train mode.
        """
        super().train(mode)
        if mode:
            # If we are going into training mode, the weights will change,
            # so we must invalidate the cache.
            self._normalized_embeddings_cache = None
            print("--- Switched to train mode. Cache invalidated. ---")
        return self

    def eval(self):
        """
        Overrides the default eval method to pre-calculate and cache the
        normalized embeddings.
        """
        super().eval()
        # "Pre-warm" the cache as soon as we switch to evaluation mode.
        print("--- Switched to eval mode. Pre-warming cache... ---")
        self._get_normalized_embeddings()
        return self

    # --- Caching and Utility Methods ---

    def _normalize_vectors(self, vectors):
        """Helper function to L2-normalize a tensor of vectors."""
        return F.normalize(vectors, p=2, dim=-1)

    def _get_normalized_embeddings(self):
        """
        Retrieves the cached normalized embeddings.
        If the cache is empty, it calculates, caches, and returns them.
        """
        if self._normalized_embeddings_cache is None:
            print("--- Calculating and caching normalized embeddings... ---")
            all_embeddings = self.in_embeddings.weight.detach().cpu()
            self._normalized_embeddings_cache = self._normalize_vectors(all_embeddings)
        else:
            print("--- Using cached embeddings. ---")
        
        return self._normalized_embeddings_cache

    def get_vector(self, word, missing='zeros'):
        # ... (this method remains the same) ...
        if word in self.word_to_ix:
            word_index = torch.tensor([self.word_to_ix[word]])
            return self.in_embeddings(word_index).detach().cpu().squeeze()
        else:
            if missing == 'random': return torch.randn(self.embedding_dim)
            else: return torch.zeros(self.embedding_dim)

    def get_sentence_vector(self, sentence, missing='zeros'):
        # ... (this method remains the same) ...
        words = sentence.split()
        vectors = [self.get_vector(word, missing) for word in words]
        valid_vectors = [v for v in vectors if v is not None]
        if not valid_vectors: return torch.zeros(self.embedding_dim)
        sentence_tensor = torch.stack(valid_vectors)
        mean_vector = torch.mean(sentence_tensor, dim=0)
        return self._normalize_vectors(mean_vector)

    # --- Similarity Methods Using the Cache ---

    def find_most_similar(self, word, top_n=10):
        # Calling self.eval() is no longer strictly necessary here because we assume
        # the user does it once before starting inference, but it's good practice.
        if self.training: self.eval()
        input_vector = self.get_vector(word)
        if torch.all(input_vector.eq(0)): return []
        return self.find_most_similar_by_vector(input_vector, top_n=top_n, exclude_word=word)

    def find_most_similar_by_vector(self, vector, top_n=10, exclude_word=None):
        if self.training: self.eval()
        all_embeddings_normalized = self._get_normalized_embeddings()
        input_vector_normalized = self._normalize_vectors(vector)
        similarities = torch.matmul(all_embeddings_normalized, input_vector_normalized)
        k = top_n + 1 if exclude_word else top_n
        top_results = torch.topk(similarities, k=k)
        
        results = []
        exclude_index = self.word_to_ix.get(exclude_word, -1)
        for score, idx in zip(top_results.values, top_results.indices):
            if idx.item() == exclude_index: continue
            if len(results) < top_n:
                results.append((self.ix_to_word[idx.item()], score.item()))
        return results
        
    def save_checkpoint(self, path, optimizer=None, epoch=None):
        """
        Saves a checkpoint of the model and associated metadata.

        Args:
            path (str): The path to save the checkpoint file.
            optimizer (torch.optim.Optimizer, optional): The optimizer state to save. Defaults to None.
            epoch (int, optional): The epoch number to save. Defaults to None.
        """
        
        checkpoint = {
            'model_state_dict': self.module.state_dict() if isinstance(self, nn.DataParallel) else self.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'word_to_ix': self.word_to_ix,
            'ix_to_word': self.ix_to_word
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        torch.save(checkpoint, path)
        print(f"Model checkpoint saved successfully to '{path}'.")


    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device=torch.device("cpu")):
        """
        Loads a model from a checkpoint file.
        This is a factory method that creates a new model instance.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        Returns:
            SkipGramNegSampling: The loaded model instance.
        """
        # 1. Load the checkpoint dictionary
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 2. Instantiate the model using the saved hyperparameters and mappings
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            word_to_ix=checkpoint['word_to_ix'],
            ix_to_word=checkpoint['ix_to_word']
        )

        # 3. Load the model's learned weights
        model.load_state_dict(checkpoint['model_state_dict'])
                
        return model