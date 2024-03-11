import dfjss_nn

autoencoder_folder = "Autoencoder input-200-100-200-output"
autoencoder_state_path = f"{autoencoder_folder}/model_epoch279.pth"

reward_model_folder = "Reward Model Seed 50 6x256"
reward_model_state_path = f"{reward_model_folder}/model_epoch3.pth"

sequence_length = 32
tree_depth = 4

autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=sequence_length)

autoencoder.import_state(autoencoder_state_path)

print(autoencoder.summary())

dataset = dfjss_nn.RewardModelDataset(autoencoder=autoencoder,
                                      autoencoder_folder=autoencoder_folder,
                                      anti_decode=True)

print(dataset.summary())

seeds = [50]
if seeds is None:
    seeds = dataset.seeds

embedding_dim = -1 if seeds is not None and len(seeds) == 1 else 256

reward_model = dfjss_nn.RewardModel(input_size=autoencoder.encoding_size,
                                    seeds=seeds,
                                    embedding_dim=embedding_dim)

reward_model.import_state(reward_model_state_path)

print(reward_model.summary())

result = dfjss_nn.optimize_reward(reward_model, autoencoder, dataset,
                                  tree_depth=tree_depth,
                                  seed_to_optimize=50,
                                  init="best_of_dataset", mode="sa", max_iters=50000, iters_per_print=2500)
