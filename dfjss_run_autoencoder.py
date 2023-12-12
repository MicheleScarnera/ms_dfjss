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
autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=dataset.max_sequence_length())

try:
    autoencoder.eval()
    autoencoded = autoencoder(example).detach()
    print(f"Autoencoded (Untrained): {dfjss_nn.string_from_onehots(autoencoded)}")
    # print(autoencoded)
    print(f"Output size: {autoencoded.size()}")
finally:
    print(autoencoder.summary())

dfjss_nn.train_autoencoder(autoencoder,
                           batch_size=64,
                           max_depth=dataset.max_depth,
                           num_epochs=500,
                           encoder_only_epochs=50,
                           train_autoencoder_size=16384,
                           train_encoder_size_percent=0.125,
                           train_encoder_size_mutated=6,
                           val_autoencoder_size=16384,
                           val_encoder_size_percent=0.125,
                           val_encoder_size_mutated=6,
                           encoder_sets_of_features_size=500,
                           flatten_trees=flatten_trees,
                           fill_trees=fill_trees)
