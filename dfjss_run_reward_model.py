import dfjss_nn

autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=32)

autoencoder.import_state(r"AUTOENCODER FEEDFORWARD THE GOOD ONE\model_epoch376.pth")

dataset = dfjss_nn.RewardModelDataset(autoencoder=autoencoder)

reward_model = dfjss_nn.RewardModel(input_size=autoencoder.encoding_size,
                                    seed_to_index=dataset.seed_to_index)

print(reward_model.summary())

dfjss_nn.train_reward_model(reward_model, dataset=dataset)
