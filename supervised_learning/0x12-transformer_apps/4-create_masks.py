#!/usr/bin/env python3
"""
MASK
"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    masks for training/validation
    """
    seq_len_in = inputs.shape[1]
    batch_size, seq_len_out = target.shape

    enc_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)[
        :, tf.newaxis, tf.newaxis, :]
    dec_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)[
        :, tf.newaxis, tf.newaxis, :]

    LAM = 1 - \
        tf.linalg.band_part(tf.ones(shape=(
            batch_size, 1, seq_len_out, seq_len_out)), -1, 0)

    dec_tpm = tf.cast(tf.math.equal(target, 0), tf.float32)[
        :, tf.newaxis, tf.newaxis, :]
    comb_mask = tf.maximum(dec_tpm, LAM)
    return enc_mask, comb_mask, dec_mask
