import dfjss_plotting as plotting

paths = ["Autoencoder input-200-100-200-output", "Autoencoder input-2400-1200-2400-output"]

for path in paths:
    print(f"Plotting \"{path}\"")
    plotting.plot_autoencoder_training(path)
