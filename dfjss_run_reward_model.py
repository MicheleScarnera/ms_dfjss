import dfjss_nn

autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=32)

folder = "Autoencoder input-200-100-200-output"
state_path = f"{folder}/model_epoch279.pth"

autoencoder.import_state(state_path)

force_seeds = [50]
embedding_dim = 256

dataset = dfjss_nn.RewardModelDataset(autoencoder=autoencoder,
                                      autoencoder_folder=folder,
                                      anti_decode=True,
                                      force_seeds=force_seeds)

reward_model = dfjss_nn.RewardModel(input_size=autoencoder.encoding_size,
                                    seeds=dataset.seeds,
                                    embedding_dim=-1 if force_seeds is not None and len(force_seeds) == 1 else embedding_dim,
                                    num_layers=6,
                                    layer_widths=(256,),
                                    layers_are_residual=True,
                                    layer_dropout=0.)

print(dataset.summary())

print(reward_model.summary())

dfjss_nn.train_reward_model(reward_model, dataset=dataset, weight_decay=0., loss_weights=(0., 1., 0.), num_epochs=200)
