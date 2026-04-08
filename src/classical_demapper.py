import tensorflow as tf
from sionna.phy.mapping import Demapper

def create_classical_demapper(bits_per_symbol):
    return Demapper("app", "qam", num_bits_per_symbol=bits_per_symbol)

def classical_demapper_call(demapper, y, noise):
    llr = demapper(y, noise)
    return tf.reshape(llr, [tf.shape(y)[0], -1])
