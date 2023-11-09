import dfjss_nn

autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=32)

autoencoder.import_state(r"AUTOENCODER FEEDFORWARD 2023-11-01 17-04-22\model_epoch376.pth")

dataset = dfjss_nn.RewardModelDataset()

reward_model = dfjss_nn.RewardModel(input_size=autoencoder.encoding_size,
                                    num_rewards=dataset.num_rewards)

print(reward_model.summary())

dfjss_nn.train_reward_model(reward_model, autoencoder, dataset=dataset)
