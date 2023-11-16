import dfjss_nn

autoencoder_type = "FEEDFORWARD"

flatten_trees = False
fill_trees = False

# some architectures can only work with specific settings
if autoencoder_type == "FEEDFORWARD":
    fill_trees = True

# creating a dataset to print one example, and get some parameters needed for later
dataset = dfjss_nn.AutoencoderDataset(size=1, flatten_trees=flatten_trees, fill_trees=fill_trees)

for datapoint in dataset:
    print(f"Original: {dfjss_nn.string_from_onehots(datapoint)}")
    print(f"Reduced: {dfjss_nn.string_from_onehots(dfjss_nn.reduce_sequence(datapoint), vocab=dfjss_nn.VOCAB_REDUCED)}")
    #print(datapoint)
    print(f"Input size: {datapoint.size()}")
    example = datapoint
    break

# create the untrained model
if autoencoder_type == "RNN":
    autoencoder = dfjss_nn.IndividualRNNAutoEncoder()
elif autoencoder_type == "TRANSFORMER":
    autoencoder = dfjss_nn.IndividualTransformerAutoEncoder(max_length=dataset.max_sequence_length())
elif autoencoder_type == "FEEDFORWARD":
    autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=dataset.max_sequence_length())
else:
    raise Exception("autoencoder_type unknown")

try:
    autoencoder.eval()
    autoencoded = autoencoder(example).detach()
    print(f"Autoencoded (Untrained): {dfjss_nn.string_from_onehots(autoencoded)}")
    #print(autoencoded)
    print(f"Output size: {autoencoded.size()}")
finally:
    print(autoencoder.summary())

dfjss_nn.train_autoencoder(autoencoder,
                           batch_size=64,
                           max_depth=dataset.max_depth,
                           num_epochs=500,
                           train_size=16384,
                           val_size=16384,
                           flatten_trees=flatten_trees,
                           fill_trees=fill_trees)
