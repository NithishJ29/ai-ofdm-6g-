import matplotlib.pyplot as plt
from sionna.phy.mapping import Mapper

from src.utils import (
    get_simulation_params,
    train_ai_demapper,
    run_simulation
)
from src.channel import create_resource_grid, create_channel, create_mappers
from src.classical_demapper import create_classical_demapper
from src.ai_demapper import create_nn_demapper
from src.ldpc import create_ldpc

# ======================
# Setup
# ======================
params = get_simulation_params()

rg = create_resource_grid()
channel, estimator = create_channel(rg)
rg_mapper, rg_demapper = create_mappers(rg)

mapper = Mapper("qam", params["num_bits_per_symbol"])
demapper_classical = create_classical_demapper(params["num_bits_per_symbol"])

num_symbols = rg.num_data_symbols
total_bits = num_symbols * params["num_bits_per_symbol"]

encoder, decoder = create_ldpc(512, 1024)

nn_model = create_nn_demapper((num_symbols, 5), params["num_bits_per_symbol"])

# ======================
# Train
# ======================
print("Training AI model...")
train_ai_demapper(
    nn_model, mapper, rg_mapper, rg_demapper,
    channel, estimator, demapper_classical,
    num_symbols, params["batch_size"]
)

# ======================
# Simulate
# ======================
print("\nRunning simulation...")
ber_classical, ber_ai = run_simulation(
    params, mapper, rg_mapper, rg_demapper,
    channel, estimator,
    demapper_classical, nn_model,
    encoder, decoder,
    num_symbols, total_bits
)

# ======================
# Plot
# ======================
plt.semilogy(params["snr_range_db"], ber_classical, 'o-', label="Classical")
plt.semilogy(params["snr_range_db"], ber_ai, 's-', label="AI")
plt.legend()
plt.grid(True)
plt.savefig("ber_vs_snr.png")

print("\nDone. Plot saved.")