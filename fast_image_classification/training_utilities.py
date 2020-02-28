from collections import namedtuple
from pathlib import Path
from random import shuffle
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from fast_image_classification.preprocessing_utilities import (
    read_img_from_path,
    resize_img,
)

SampleFromPath = namedtuple("Sample", ["path", "target_vector"])
import imgaug.augmenters as iaa


def chunks(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def get_seq():
    sometimes = lambda aug: iaa.Sometimes(0.1, aug)
    seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(scale={"x": (0.8, 1.2)})),
            sometimes(iaa.Fliplr(p=0.5)),
            sometimes(iaa.Affine(scale={"y": (0.8, 1.2)})),
            sometimes(iaa.Affine(translate_percent={"x": (-0.2, 0.2)})),
            sometimes(iaa.Affine(translate_percent={"y": (-0.2, 0.2)})),
            sometimes(iaa.Affine(rotate=(-20, 20))),
            sometimes(iaa.Affine(shear=(-20, 20))),
            sometimes(iaa.AdditiveGaussianNoise(scale=0.07 * 255)),
            sometimes(iaa.GaussianBlur(sigma=(0, 3.0))),
        ],
        random_order=True,
    )
    return seq


def batch_generator(
    list_samples,
    batch_size=32,
    pre_processing_function=None,
    resize_size=(128, 128),
    augment=False,
):
    seq = get_seq()
    pre_processing_function = (
        pre_processing_function
        if pre_processing_function is not None
        else preprocess_input
    )
    while True:
        shuffle(list_samples)
        for batch_samples in chunks(list_samples, size=batch_size):
            images = [read_img_from_path(sample.path) for sample in batch_samples]

            if augment:
                images = seq.augment_images(images=images)

            images = [resize_img(x, h=resize_size[0], w=resize_size[1]) for x in images]

            images = [pre_processing_function(a) for a in images]
            targets = [sample.target_vector for sample in batch_samples]
            X = np.array(images)
            Y = np.array(targets)

            yield X, Y


def dataframe_to_list_samples(df, binary_targets, base_path, image_name_col):
    paths = df[image_name_col].apply(lambda x: str(Path(base_path) / x)).tolist()
    targets = df[binary_targets].values.tolist()

    samples = [
        SampleFromPath(path=path, target_vector=target_vector)
        for path, target_vector in zip(paths, targets)
    ]

    return samples
