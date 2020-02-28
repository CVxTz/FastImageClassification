import argparse

import numpy as np
import yaml
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from fast_image_classification.preprocessing_utilities import (
    read_img_from_path,
    resize_img,
    read_from_file,
)
from fast_image_classification.utils import download_model


class ImagePredictor:
    def __init__(
        self, model_path, resize_size, targets, pre_processing_function=preprocess_input
    ):
        self.model_path = model_path
        self.pre_processing_function = pre_processing_function
        self.model = load_model(self.model_path)
        self.resize_size = resize_size
        self.targets = targets

    @classmethod
    def init_from_config_path(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        predictor = cls(
            model_path=config["model_path"],
            resize_size=config["resize_shape"],
            targets=config["targets"],
        )
        return predictor

    @classmethod
    def init_from_config_url(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)

        download_model(
            config["model_url"], config["model_path"], config["model_sha256"]
        )

        return cls.init_from_config_path(config_path)

    def predict_from_array(self, arr):
        arr = resize_img(arr, h=self.resize_size[0], w=self.resize_size[1])
        arr = self.pre_processing_function(arr)
        pred = self.model.predict(arr[np.newaxis, ...]).ravel().tolist()
        pred = [round(x, 3) for x in pred]
        return {k: v for k, v in zip(self.targets, pred)}

    def predict_from_path(self, path):
        arr = read_img_from_path(path)
        return self.predict_from_array(arr)

    def predict_from_file(self, file_object):
        arr = read_from_file(file_object)
        return self.predict_from_array(arr)


if __name__ == "__main__":
    """
    python train.py --predictor_config "../example/predictor_config.yaml"

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictor_config_path",
        help="predictor_config_path",
        default="../example/predictor_config.yaml",
    )

    args = parser.parse_args()

    predictor_config_path = args.predictor_config_path

    predictor = ImagePredictor.init_from_config_path(predictor_config_path)

    # pred = predictor.predict_from_path(
    #     "../example/data/0a42098532a047e0b20f8dc22a9c1d6f.jpg"
    # )
    # print(pred)
    #
    # pred = predictor.predict_from_path(
    #     "../example/data/0ab5f5ba8484483a8f8003c0152e181d.jpg"
    # )
    # print(pred)
    #
    # with open("../example/data/0ab5f5ba8484483a8f8003c0152e181d.jpg", "rb") as f:
    #     _pred = predictor.predict_from_file(f)
    #
    # print(_pred)
