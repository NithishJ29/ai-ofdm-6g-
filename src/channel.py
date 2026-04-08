import tensorflow as tf
import numpy as np
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, ResourceGridDemapper, LSChannelEstimator
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel import OFDMChannel, RayleighBlockFading

def create_resource_grid():
    return ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=64,
        subcarrier_spacing=15000,
        num_tx=1,
        num_streams_per_tx=1,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[0],
    )

def create_channel(rg):
    channel_model = RayleighBlockFading(
        num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1
    )
    channel = OFDMChannel(
        channel_model=channel_model,
        resource_grid=rg,
        add_awgn=True,
        return_channel=True
    )
    estimator = LSChannelEstimator(rg, interpolation_type="nn")
    return channel, estimator

def create_mappers(rg):
    rg_mapper = ResourceGridMapper(rg)
    stream_management = StreamManagement(np.ones([1, 1]), 1)
    rg_demapper = ResourceGridDemapper(rg, stream_management)
    return rg_mapper, rg_demapper
