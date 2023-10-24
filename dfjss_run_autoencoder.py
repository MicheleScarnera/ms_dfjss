import dfjss_nn

autoencoder_type = "FEEDFORWARD"

dataset = dfjss_nn.AutoencoderDataset()

for datapoint in dataset:
    print(f"Original: {dfjss_nn.string_from_onehots(datapoint)}")
    print(f"Reduced: {dfjss_nn.string_from_onehots(dfjss_nn.reduce_sequence(datapoint), vocab=dfjss_nn.VOCAB_REDUCED)}")
    #print(datapoint)
    print(f"Input size: {datapoint.size()}")
    example = datapoint
    break

if autoencoder_type == "RNN":
    autoencoder = dfjss_nn.IndividualRNNAutoEncoder()
elif autoencoder_type == "TRANSFORMER":
    autoencoder = dfjss_nn.IndividualTransformerAutoEncoder(max_length=dataset.max_length())
elif autoencoder_type == "FEEDFORWARD":
    autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=dataset.max_length())
else:
    raise Exception("autoencoder_type unknown")

autoencoder.eval()
autoencoded = autoencoder(example).detach()
print(f"Autoencoded (Untrained): {dfjss_nn.string_from_onehots(autoencoded)}")
#print(autoencoded)
print(f"Output size: {autoencoded.size()}")

print(autoencoder.summary())

dfjss_nn.train_autoencoder(autoencoder,
                           batch_size=64,
                           max_depth=dataset.max_depth,
                           num_epochs=200,
                           train_size=8192,
                           val_size=2048,
                           regularization_coefficient=10.)
