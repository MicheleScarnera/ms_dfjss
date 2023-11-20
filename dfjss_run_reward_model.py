import dfjss_nn

autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=32)

autoencoder.import_state(r"AUTOENCODER FEEDFORWARD THE GOOD ONE\model_epoch376.pth")

dataset = dfjss_nn.RewardModelDataset(autoencoder=autoencoder)

reward_model = dfjss_nn.RewardModel(input_size=autoencoder.encoding_size,
                                    seed_to_index=dataset.seed_to_index,
                                    embedding_dim=336,
                                    num_layers=2,
                                    residual_layers=True,
                                    layer_dropout=0.1,
                                    num_obscured_seeds=2)

print(reward_model.summary())

dfjss_nn.train_reward_model(reward_model, dataset=dataset, weight_decay=0., loss_weights=(0., 1., 0.))
