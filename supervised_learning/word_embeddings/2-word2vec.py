#!/usr/bin/env python3
"""This modlue creates a word2vec model"""
from gensim.models import Word2Vec


def word2vec_model(
    sentences,
    vector_size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1,
):
    """This function creates builds and trains a word2vec model
    Args:
        sentences is a list of sentences to be trained on
        vector_size is the dimensionality of the embedding layer
        min_count is the minimum number of occurrences of a word
        for use in training
        window is the maximum distance between the current and
        predicted word within a sentence
        negative is the size of negative sampling
        cbow is a boolean to determine the training type; True is
        for CBOW; False is for Skip-gram
        epochs is the number of iterations to train over
        seed is the seed for the random number generator
        workers is the number of worker threads to train the model
    Returtns:
        the trained model
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=cbow,
        seed=seed,
        workers=workers,
        epochs=epochs,
    )

    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
