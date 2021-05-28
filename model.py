"""
:Author: Twardosz Adam (hbery@github.com)
"""

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
            name="conv2d_64",
            input_shape=(224, 224, 3),
    ))
    # 2nd layer
    # MaxPooling2D output data 

    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool2D_64"))

    # 3rd layer
    # Convolution2D output data 
    model.add(layers.Convolution2D( 128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            name="conv2d_128"
    ))
    # 4th layer
    # MaxPooling2D output data 

    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2d_128"))

    # 5th layer
    # Convolution2D output data 
    model.add(layers.Convolution2D( 256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            name="conv2d_256"
    ))
    # 6th layer
    # MaxPooling2D output data 

    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2d_256"))

    # 7th layer
    # Convolution2D output data 
    model.add(layers.Convolution2D( 512,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            name="conv2d_512"
    ))
    # 8th layer
    # MaxPooling2D output data 

    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2d_512"))

    # 9th layer
    # flatten output data 
    model.add(layers.Flatten(name="flatten"))

    # 10th layer
    model.add(layers.Dense( 4096, activation='relu', name="fc_4096"))

    # 11th layer
    model.add(layers.Dense( 128, activation='relu', name="fc_128"))

    # 12th layer
    model.add(layers.Dense( length, name=f"fc_{length}" ))

    # 13th layer
    model.add(layers.Softmax(name="softmax"))

    # Compile model
    model.compile(optimizer='adam',
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = build_model(6, name="MODEL_BIQA")
    print(model.summary())    
    