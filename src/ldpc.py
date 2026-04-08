
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder

def create_ldpc(k, n):
    encoder = LDPC5GEncoder(k=k, n=n)
    decoder = LDPC5GDecoder(encoder)
    return encoder, decoder