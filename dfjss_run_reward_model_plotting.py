import dfjss_plotting as plotting

paths = ["Reward Model All Seeds 6x256", "Reward Model Seed 50 6x256"]
baselines = [(162.9, None, 140848.33), (23.81, None, 1566.28)]

for path, baseline in zip(paths, baselines):
    print(f"Plotting \"{path}\"")
    plotting.plot_reward_model_training(path, baseline)
