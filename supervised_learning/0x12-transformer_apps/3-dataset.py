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

    def __init__(self, batch_size, max_len):
        """
        init
        """
        (self.data_train, self.data_valid), meta = tfds.load(
            'ted_hrlr_translate/pt_to_en', split=['train', 'validation'],
            as_supervised=True, with_info=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.data_train = self.data_train.map(self.tf_encode)
        num = meta.splits['train'].num_examples

        self.data_train = self.data_train.filter(
            lambda x, y: tf.math.logical_and(
                tf.size(x) <= max_len, tf.size(y) <= max_len))
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(num).padded_batch(
            batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(
            lambda i, j: tf.math.logical_and(
                tf.size(i) <= max_len, tf.size(j) <= max_len))
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None]))

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

    def encode(self, pt, en):
        """
        Encode
        """
        return [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8')) + [self.tokenizer_pt.vocab_size + 1], [
            self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
                en.numpy().decode('utf-8')) + [
                    self.tokenizer_en.vocab_size + 1]

    def tf_encode(self, pt, en):
        """
        Tensorflow wrapper
        """
        tensor_pt, tensor_en = tf.py_function(
            self.encode, inp=[
                pt, en], Tout=[
                tf.int64, tf.int64])
        tensor_pt.set_shape([None])
        tensor_en.set_shape([None])
        return tensor_pt, tensor_en
