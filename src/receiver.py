import tensorflow as tf

def mmse_equalize(y, h, noise, err_var=None):
    h_conj = tf.math.conj(h)
    h_power = tf.abs(h) ** 2

    if err_var is not None:
        denom = h_power + noise + err_var
    else:
        denom = h_power + noise

    eq = (h_conj / tf.cast(denom, h_conj.dtype)) * y
    noise_eff = noise / denom

    return eq, noise_eff
