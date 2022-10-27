#!/usr/bin/env python3
"""
PREPROCESS DATA
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as plt


class WindowGenerator():
    """generates window"""

    def __init__(self, input_width, label_width, shift,
                 train_df,
                 val_df,
                 test_df,
                 label_columns=None):
        """init"""
        """Store the raw data."""
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        """Work out the label column indices."""
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        """Work out the window parameters."""
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        """repr"""
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        """split window"""
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]]
                               for name in self.label_columns], axis=-1)

        """Slicing doesn't preserve static shape information, so set the shapes
            manually. This way the `tf.data.Datasets` are easier to inspect."""
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """make dataset"""
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        """returns the training dataset"""
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """returns the validation dataset"""
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """returns the testing dataset"""
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """example"""
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            """No example batch was found, so get one from
             the `.train` dataset"""
            result = next(iter(self.train))
            """And cache it for next time"""
            self._example = result
        return result


def preprocess():
    """preprocess data"""
    df = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
    df.drop(['Open',
             'High',
             'Low',
             'Volume_(BTC)',
             'Volume_(Currency)',
             'Weighted_Price'],
            axis=1,
            inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.columns = ['Time', 'Closing Price']
    df = df[df['Time'] >= '2018']
    df.dropna(inplace=True)
    df = df.set_index('Time').asfreq('1H')
    df.dropna(inplace=True)
    df = (df - df.mean()) / df.std()
    n = len(df)
    training_set = df[0:int(n * 0.7)]
    validation_set = df[int(n * 0.7):int(n * 0.9)]
    testing_set = df[int(n * 0.9):]
    window = WindowGenerator(
        input_width=24,
        label_width=1,
        shift=1,
        label_columns=['Closing Price'],
        train_df=training_set,
        val_df=validation_set,
        test_df=testing_set)
    return(window)
