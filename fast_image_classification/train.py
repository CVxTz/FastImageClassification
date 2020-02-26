import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from training_utilities import dataframe_to_list_samples, batch_generator
from models import get_model_classification


def train_from_csv(csv_path, data_config_path, training_config_path):
    df = pd.read_csv(csv_path)
    train, val = train_test_split(df, test_size=0.2, random_state=1337)

    with open(data_config_path, "r") as f:
        data_config = yaml.load(f, yaml.SafeLoader)
    with open(training_config_path, "r") as f:
        training_config = yaml.load(f, yaml.SafeLoader)

    train_samples = dataframe_to_list_samples(
        train,
        binary_targets=data_config["targets"],
        base_path=data_config["images_base_path"],
        image_name_col=data_config["image_name_col"],
    )
    val_samples = dataframe_to_list_samples(
        val,
        binary_targets=data_config["targets"],
        base_path=data_config["images_base_path"],
        image_name_col=data_config["image_name_col"],
    )

    model = get_model_classification(
        input_shape=tuple(data_config["input_shape"]),
        n_classes=len(data_config["targets"]),
    )
    train_gen = batch_generator(
        train_samples,
        resize_size=data_config["resize_shape"],
        augment=training_config["use_augmentation"],
    )
    val_gen = batch_generator(val_samples, resize_size=data_config["resize_shape"])

    model.fit_generator(
        train_gen,
        steps_per_epoch=len(train_samples) // training_config["batch_size"],
        validation_data=val_gen,
        validation_steps=len(val_samples) // training_config["batch_size"],
        epochs=training_config["epochs"],
    )


if __name__ == "__main__":
    csv_path = "../example/data.csv"
    data_config_path = "../example/data_config.yaml"
    training_config_path = "../example/training_config.yaml"
    train_from_csv(
        csv_path=csv_path,
        data_config_path=data_config_path,
        training_config_path=training_config_path,
    )
