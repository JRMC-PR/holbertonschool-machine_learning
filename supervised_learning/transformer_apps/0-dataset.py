#!/usr/bin/env python3
"""This module contains the class Dataset
   that loads and prepares a dataset for machine translation."""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """This class loads and prepares a dataset for machine translation."""

    def __init__(self):
        """Class constructor
        self.data_train: contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset train split, loaded as_supervised
        self.data_valid: contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset validate split, loaded as_supervised
        self.tokenizer_pt: Portuguese tokenizer created from the training set
        self.tokenizer_en: English tokenizer created from the training set
        """
        # Load the training and validation datasets
        self.data_train = tfds.load(
            "ted_hrlr_translate/pt_to_en", split="train", as_supervised=True
        )
        self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="validation", as_supervised=True
        )

        # Tokenize the datasets
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset.
        Args:
            data (tf.data.Dataset): dataset whose examples are
            formatted as a tuple (pt, en)
                pt (tf.Tensor): contains the Portuguese sentence
                en (tf.Tensor): contains the corresponding English sentence
        Returns:
            tokenizer_pt (BertTokenizer): Portuguese tokenizer
            tokenizer_en (BertTokenizer): English tokenizer
        """
        # Load pre-trained tokenizers
        tokenizer_pt = transformers.BertTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased")

        # Train the tokenizers with a maximum vocabulary size of 2**13
        tokenizer_pt.add_tokens(
            [pt.numpy().decode("utf-8") for pt, en in data])
        tokenizer_en.add_tokens(
            [en.numpy().decode("utf-8") for pt, en in data])

        return tokenizer_pt, tokenizer_en
