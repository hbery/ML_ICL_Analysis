#!/usr/bin/env python3

"""Model builder ( full images )

    :Date: 04.2021
    :Author: Adam Twardosz (a.twardosz98@gmail.com, https://github.com/hbery)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras import layers, models, losses


def build_model(length: int, *, name: str="ConvModel") -> models.Sequential:
    """Build model for IMAGE COMPRESSION LEVEL DETECTION 
    -> function very specific to our problem

    :param length: Number of categories
    :type length: int
    :param name: Name of the model, defaults to "ConvModel"
    :type name: str, optional
    
    :return: Built model
    :rtype: tf.keras.models.Sequential
    """
    
    model = models.Sequential(name=name)

    # 1st layer
    # Convolution2D output data
    model.add(layers.Convolution2D( 64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            name="cv2d_64",
            input_shape=(224, 224, 3),
    ))

    # 2nd layer
    # MaxPooling2D output data 
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="mp2d_64"))

    # 3rd layer
    # Convolution2D output data 
    model.add(layers.Convolution2D( 128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            name="cv2d_128"
    ))

    # 4th layer
    # MaxPooling2D output data 
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="mp2d_128"))

    # 5th layer
    # Convolution2D output data 
    model.add(layers.Convolution2D( 256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            name="cv2d_256"
    ))

    # 6th layer
    # MaxPooling2D output data 
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="mp2d_256"))

    # 7th layer
    # Convolution2D output data 
    model.add(layers.Convolution2D( 512,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            name="cv2d_512"
    ))

    # 8th layer
    # MaxPooling2D output data 
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="mp2d_512"))

    # 9th layer
    # Convolution2D output data 
    model.add(layers.Convolution2D( 1024,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            name="cv2d_1024"
    ))

    # 10th layer
    # MaxPooling2D output data
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="mp2d_1024"))

    # 11th layer
    # flatten output data 
    model.add(layers.Flatten(name="flatten"))

    # 12th layer
    model.add(layers.Dense( 4096, activation='relu', name="fc_4096"))

    # 13th layer
    model.add(layers.Dense( 128, activation='relu', name="fc_128"))

    # 14th layer
    model.add(layers.Dense( length, name=f"fc_{length}" ))

    # Compile model
    model.compile(optimizer='adam',
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
    )

    return model


""" MAIN """
if __name__ == "__main__":
    model = build_model(6, name="MODEL_BIQA")
    print(model.summary())    
    