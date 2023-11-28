import dfjss_nn

autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=32)

folder = "AUTOENCODER FEEDFORWARD THE GOOD ONE"
state_path = f"{folder}/model_epoch376.pth"

autoencoder.import_state(state_path)

dataset = dfjss_nn.RewardModelDataset(autoencoder=autoencoder,
                                      autoencoder_folder=folder,
                                      anti_decode=True,
                                      force_num_nonmean_seeds=0)

reward_model = dfjss_nn.RewardModel(input_size=autoencoder.encoding_size,
                                    seeds=dataset.seeds,
                                    embedding_dim=128,
                                    num_layers=2,
                                    layers_are_residual=True,
                                    layer_dropout=0.1)

print(dataset.summary())

print(reward_model.summary())

dfjss_nn.train_reward_model(reward_model, dataset=dataset, weight_decay=0., loss_weights=(0., 0., 1.), num_epochs=200)
