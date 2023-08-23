import tensorflow as tf

def image_encoder(output_layer = 16 , rgb = False):
    img_height, img_width = 135, 180
    if rgb:
        input_shape = (img_height, img_width,3)
    else:
        input_shape = (img_height, img_width,1)

    # Keras Sequential class initialization
    encoder = tf.keras.models.Sequential()

    # Add convolutional layers to extract features from the input image
    encoder.add(tf.keras.layers.Conv2D(32, (7, 7), activation='relu', input_shape=input_shape))
    encoder.add(tf.keras.layers.MaxPooling2D((2, 2)))
    encoder.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    encoder.add(tf.keras.layers.MaxPooling2D((2, 2)))
    encoder.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    encoder.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # Flatten the output of the convolutional layers
    encoder.add(tf.keras.layers.Flatten())

    # Add a fully connected layer with n outputs
    encoder.add(tf.keras.layers.Dense(output_layer, activation='linear'))
    encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mean_squared_error',metrics=['accuracy','mae','mse'])

    return encoder

