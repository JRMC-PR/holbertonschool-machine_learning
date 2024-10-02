#!/usr/bin/env python3
"""This module conmtains the class Dataset
    That loads and prepares a dataset for machine translation """
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """This class loads and prepares a dataset for machine translation"""
    def __init__(self):
        """Class constructor
            self.data_train: contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset train split, loaded as_supervised
            self.data_valid: contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset validate split, loaded as_supervised
            self.tokenizer_pt: Portuguese tokenizer created from the training
            set
            self.tokenizer_en: English tokenizer created from the training set
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset
        Args:
            data: tf.data.Dataset whose examples are formatted as a
            tuple (pt, en)
                pt: tf.Tensor containing the Portuguese sentence
                en: tf.Tensor containing the corresponding English sentence
        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        tokenizer_pt = transformers.BertTokenizer.from_pretrained(
            'bert-base-uncased',
            lower=True
        )
        tokenizer_en = transformers.BertTokenizer.from_pretrained(
            'bert-base-uncased',
            lower=True
        )
        return tokenizer_pt, tokenizer_en
