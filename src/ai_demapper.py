import tensorflow as tf

def create_nn_demapper(input_shape, num_bits_per_symbol):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_bits_per_symbol)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )
    return model

def nn_demapper_call(model, y, noise, h, num_symbols):
    y_real = tf.math.real(y)
    y_imag = tf.math.imag(y)
    h_real = tf.math.real(h)
    h_imag = tf.math.imag(h)

    nn_input = tf.stack([y_real, y_imag, noise, h_real, h_imag], axis=-1)
    nn_input = tf.reshape(nn_input, [tf.shape(y)[0], num_symbols, 5])

    llr = model(nn_input)
    return tf.reshape(llr, [tf.shape(y)[0], -1])
