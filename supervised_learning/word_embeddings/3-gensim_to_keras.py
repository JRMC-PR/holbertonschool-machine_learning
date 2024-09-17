#!/usr/bin/env python3
"""This module has the method gensim_to_keras"""


def gensim_to_keras(model):
    """This method converts a gensim word2vec model to a keras Embedding layer
    Args:
        model is a trained gensim word2vec models
    Returns:
        the trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
