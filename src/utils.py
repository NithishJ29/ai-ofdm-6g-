import tensorflow as tf
import numpy as np

def get_simulation_params():
    return {
        "batch_size": 1000,
        "num_bits_per_symbol": 2,
        "snr_range_db": np.arange(0, 21, 2),
        "num_iterations": 10
    }

def train_ai_demapper(nn_model, mapper, rg_mapper, rg_demapper,
                      channel, estimator, demapper_classical,
                      num_symbols, batch_size, epochs=5):

    train_inputs = []
    train_targets = []

    for _ in range(10):
        snr = np.random.uniform(0, 20)
        noise_var = 10 ** (-snr / 10)
        noise_tf = tf.constant(noise_var, dtype=tf.float32)

        bits = tf.random.uniform((batch_size, num_symbols*2), 0, 2, dtype=tf.int32)
        symbols = mapper(bits)
        symbols = tf.reshape(symbols, [batch_size, 1, 1, -1])

        x_rg = rg_mapper(symbols)
        y, _ = channel(x_rg, noise_tf)

        h_hat, _ = estimator(y, noise_tf)
        h_hat = tf.squeeze(h_hat, axis=[3, 4])

        # MMSE equalization
        h_conj = tf.math.conj(h_hat)
        h_power = tf.abs(h_hat)**2
        y_eq = (h_conj/tf.cast(h_power+noise_tf, h_conj.dtype)) * y

        noise_eff = noise_tf/(h_power+noise_tf)

        y_data = rg_demapper(y_eq)
        noise_data = rg_demapper(noise_eff)
        h_data = rg_demapper(h_hat)

        y_data = tf.squeeze(y_data, axis=[1,2])
        noise_data = tf.squeeze(noise_data, axis=[1,2])
        h_data = tf.squeeze(h_data, axis=[1,2])

        target = demapper_classical(y_data, noise_data)
        target = tf.reshape(target, [batch_size, num_symbols, 2])

        y_real = tf.math.real(y_data)
        y_imag = tf.math.imag(y_data)
        h_real = tf.math.real(h_data)
        h_imag = tf.math.imag(h_data)

        inp = tf.stack([y_real, y_imag, noise_data, h_real, h_imag], axis=-1)

        train_inputs.append(inp)
        train_targets.append(target)

    train_input = tf.concat(train_inputs, axis=0)
    train_target = tf.concat(train_targets, axis=0)

    nn_model.fit(train_input, train_target, epochs=epochs, batch_size=256, verbose=1)
def run_simulation(params, mapper, rg_mapper, rg_demapper,
                   channel, estimator,
                   demapper_classical, nn_model,
                   encoder, decoder,
                   num_symbols, total_bits):

    snr_range_db = params["snr_range_db"]
    batch_size = params["batch_size"]
    num_iterations = params["num_iterations"]

    ber_classical = []
    ber_ai = []

    for snr_db in snr_range_db:
        noise_var = 10 ** (-snr_db / 10)
        noise_tf = tf.constant(noise_var, dtype=tf.float32)

        ber_c = 0
        ber_a = 0

        for _ in range(num_iterations):
            bits = tf.random.uniform((batch_size, total_bits), 0, 2, dtype=tf.int32)

            symbols = mapper(bits)
            symbols = tf.reshape(symbols, [batch_size, 1, 1, -1])
            x_rg = rg_mapper(symbols)

            y, _ = channel(x_rg, noise_tf)

            h_hat, _ = estimator(y, noise_tf)
            h_hat = tf.squeeze(h_hat, axis=[3, 4])

            h_conj = tf.math.conj(h_hat)
            h_power = tf.abs(h_hat)**2

            y_eq = (h_conj/tf.cast(h_power+noise_tf, h_conj.dtype)) * y
            noise_eff = noise_tf/(h_power+noise_tf)

            y_data = rg_demapper(y_eq)
            noise_data = rg_demapper(noise_eff)
            h_data = rg_demapper(h_hat)

            y_data = tf.squeeze(y_data, axis=[1,2])
            noise_data = tf.squeeze(noise_data, axis=[1,2])
            h_data = tf.squeeze(h_data, axis=[1,2])

            # Classical
            llr_c = demapper_classical(y_data, noise_data)
            llr_c = tf.reshape(llr_c, [batch_size, total_bits])
            bits_hat_c = tf.cast(llr_c > 0, tf.int32)
            ber_c += tf.reduce_mean(tf.cast(bits != bits_hat_c, tf.float32))

            # AI
            from src.ai_demapper import nn_demapper_call
            llr_a = nn_demapper_call(nn_model, y_data, noise_data, h_data, num_symbols)
            bits_hat_a = tf.cast(llr_a > 0, tf.int32)
            ber_a += tf.reduce_mean(tf.cast(bits != bits_hat_a, tf.float32))

        ber_classical.append((ber_c/num_iterations).numpy())
        ber_ai.append((ber_a/num_iterations).numpy())

        print(f"SNR: {snr_db} dB | Classical: {ber_classical[-1]:.4f} | AI: {ber_ai[-1]:.4f}")

    return ber_classical, ber_ai