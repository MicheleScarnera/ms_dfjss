import dfjss_nn

autoencoder_folder = "AUTOENCODER FEEDFORWARD THE GOOD ONE"
autoencoder_state_path = f"{autoencoder_folder}/model_epoch376.pth"

reward_model_folder = "REWARD MODEL guinea pig"
reward_model_state_path = f"{reward_model_folder}/model_epoch2.pth"

autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=32)

autoencoder.import_state(autoencoder_state_path)

print(autoencoder.summary())

dataset = dfjss_nn.RewardModelDataset(autoencoder=autoencoder,
                                      autoencoder_folder=autoencoder_folder,
                                      anti_decode=True)

print(dataset.summary())

reward_model = dfjss_nn.RewardModel(input_size=autoencoder.encoding_size,
                                    seeds=dataset.seeds)

reward_model.import_state(reward_model_state_path)

print(reward_model.summary())

result = dfjss_nn.optimize_reward(reward_model, autoencoder, dataset,
                                  init="best_of_dataset", mode="sa", max_iters=50000, iters_per_print=1000)
