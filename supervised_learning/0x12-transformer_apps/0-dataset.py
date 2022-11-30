#!/usr/bin/env python3
"""
DATASETS
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Load/Prep Datasets
    """

    def __init__(self):
        """
        init
        """
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en', split=['train', 'validation'],
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        sub-word tokenizers
        """
        encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        tokenizer_pt = encoder([x.numpy()
                               for x, _ in data], target_vocab_size=2 ** 15)
        tokenizer_en = encoder([y.numpy()
                               for _, y in data], target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en
