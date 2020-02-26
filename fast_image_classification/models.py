from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    Concatenate,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import binary_accuracy


def get_model_classification(
    input_shape=(None, None, 3),
    model="mobilenet",
    weights="imagenet",
    n_classes=4,
    multi_class=False,
):
    inputs = Input(input_shape)
    if model == "mobilenet":
        base_model = MobileNetV2(
            include_top=False, input_shape=input_shape, weights=weights
        )
    else:
        base_model = ResNet50(
            include_top=False, input_shape=input_shape, weights=weights
        )

    x = base_model(inputs)
    x = Dropout(0.5)(x)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out = Concatenate(axis=-1)([out1, out2])
    out = Dropout(0.5)(out)
    if multi_class:
        out = Dense(n_classes, activation="softmax")(out)
    else:
        out = Dense(n_classes, activation="sigmoid")(out)

    model = Model(inputs, out)
    if multi_class:
        model.compile(
            optimizer=Adam(0.0001), loss=categorical_crossentropy, metrics=["acc"]
        )
    else:
        model.compile(
            optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=[binary_accuracy]
        )

    model.summary()

    return model
