import pandas as pd
import yaml


def train_from_csv(csv_path, data_config_path, training_config_path):
    df = pd.read_csv(csv_path)
    with open(data_config_path, "r") as f:
        data_config = yaml.load(f, yaml.SafeLoader)
    with open(training_config_path, "r") as f:
        training_config = yaml.load(f, yaml.SafeLoader)

    print(data_config)

    print(training_config)


if __name__ == "__main__":
    csv_path = "../example/data.csv"
    data_config_path = "../example/data_config.yaml"
    training_config_path = "../example/training_config.yaml"
    train_from_csv(
        csv_path=csv_path,
        data_config_path=data_config_path,
        training_config_path=training_config_path,
    )
