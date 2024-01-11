import dfjss_nn

flatten_trees = True
fill_trees = True

# creating a dataset to print one example, and get some parameters needed for later
dataset = dfjss_nn.AutoencoderDataset(size=1, flatten_trees=flatten_trees, fill_trees=fill_trees)

for datapoint in dataset:
    print(f"Original: {dfjss_nn.string_from_onehots(datapoint)}")
    print(f"Reduced: {dfjss_nn.string_from_onehots(dfjss_nn.reduce_sequence(datapoint), vocab=dfjss_nn.VOCAB_REDUCED)}")
    # print(datapoint)
    print(f"Input size: {datapoint.size()}")
    example = datapoint
    break

# create the untrained model
autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=dataset.max_sequence_length(),
                                                        encoding_size=100,
                                                        encoder_layers_widths=(200,),
                                                        decoder_layers_widths=(200,),
                                                        enable_batch_norm=False,
                                                        dropout=0.)

try:
    autoencoder.eval()
    autoencoded = autoencoder(example.unsqueeze(0)).squeeze(0).detach()
    print(f"Autoencoded (Untrained): {dfjss_nn.string_from_onehots(autoencoded)}")
    # print(autoencoded)
    print(f"Output size: {autoencoded.size()}")
finally:
    print(autoencoder.summary())

checkpoint_path = r"AUTOENCODER SHALLOW PART 1/model_epoch23.pth"

if False:
    autoencoder.import_state(checkpoint_path)
    print(f"Checkpointing from {checkpoint_path}")

dfjss_nn.train_autoencoder(autoencoder,
                           batch_size=64,
                           max_depth=dataset.max_depth,
                           num_epochs=2000,
                           encoder_only_epochs=25,
                           enable_encoder_specific_training=False,
                           train_autoencoder_size=16384,
                           train_encoder_size_percent=0.125,
                           train_encoder_size_mutated=32,
                           val_autoencoder_size=16384,
                           val_encoder_size_percent=0.125,
                           val_encoder_size_mutated=32,
                           encoder_sets_of_features_size=500,
                           flatten_trees=flatten_trees,
                           fill_trees=fill_trees)
