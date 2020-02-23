from collections import namedtuple
from random import shuffle
from pathlib import Path

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .preprocessing_utilities import read_img_from_path

SampleFromPath = namedtuple("Sample", ["path", "target_vector"])


def chunks(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def batch_generator(list_samples, batch_size=32, pre_processing_function=None):
    pre_processing_function = (
        pre_processing_function
        if pre_processing_function is not None
        else preprocess_input
    )
    while True:
        shuffle(list_samples)
        for batch_samples in chunks(list_samples, size=batch_size):
            images = [read_img_from_path(sample.path) for sample in batch_samples]
            images = [pre_processing_function(a) for a in images]
            targets = [sample.target_vector for sample in batch_samples]
            X = np.array(images)
            Y = np.array(targets)

            yield X, Y


def dataframe_to_list_samples(df, binary_targets, base_path, image_name_col):
    paths = df[image_name_col].apply(lambda x: str(Path(base_path) / x)).tolist()
    targets = df[binary_targets].tolist()

    samples = [
        SampleFromPath(path=path, target_vector=target_vector)
        for path, target_vector in zip(paths, targets)
    ]

    return samples
