import dfjss_plotting as plotting

# NO WAIT VS WAIT

plotting.plot_wait_dist("Nonwait vs Wait")

# HEAVINESS OF TAILS

plotting.plot_heaviness_dist("FIFO Cooldown 0 60", "FIFO Cooldown 0 15")

# GENETIC ALGORITHM

plotting.plot_evolution("Genetic Algorithm")

# AUTOENCODER

paths = ["Autoencoder input-200-100-200-output", "Autoencoder input-2400-1200-2400-output"]

for path in paths:
    print(f"Plotting \"{path}\"")
    plotting.plot_autoencoder_training(path)

# REWARD MODEL

paths = ["Reward Model All Seeds 6x256", "Reward Model Seed 50 6x256"]
baselines = [(162.9, None, 140848.33), (23.81, None, 1566.28)]

for path, baseline in zip(paths, baselines):
    print(f"Plotting \"{path}\"")
    plotting.plot_reward_model_training(path, baseline)
