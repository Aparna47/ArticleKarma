import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


# --- Model Definition (same as before) ---
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, word_to_ix, ix_to_word):
        super(Word2Vec, self).__init__()

        # Store mappings and parameters inside the model
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target_word_ix):
        embedded = self.in_embeddings(target_word_ix)
        scores = self.out_embeddings(embedded)
        return scores
    def find_most_similar(self, word, top_n=10):
        # ... (instance method for inference, remains the same) ...
        self.eval()
        if word not in self.word_to_ix:
            print(f"Word '{word}' not in vocabulary.")
            return []

        word_embeddings = self.out_embeddings.weight.detach().cpu()
        word_embeddings_normalized = F.normalize(word_embeddings, p=2, dim=1)
        word_index = self.word_to_ix[word]
        input_vector_normalized = word_embeddings_normalized[word_index]
        similarities = torch.matmul(word_embeddings_normalized, input_vector_normalized)
        top_results = torch.topk(similarities, k=top_n + 1)
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            if idx.item() == word_index: continue
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
    def load_from_checkpoint(cls, checkpoint_path):
        """
        Loads a model from a checkpoint file.
        This is a factory method that creates a new model instance.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        Returns:
            SkipGramNegSampling: The loaded model instance.
        """
        # 1. Load the checkpoint dictionary
        checkpoint = torch.load(checkpoint_path)

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