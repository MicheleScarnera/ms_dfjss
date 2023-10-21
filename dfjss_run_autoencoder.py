import dfjss_nn

dataset = dfjss_nn.AutoencoderDataset()

for datapoint in dataset:
    print(f"Original: {dfjss_nn.string_from_onehots(datapoint)}")
    print(f"Reduced: {dfjss_nn.string_from_onehots(dfjss_nn.reduce_sequence(datapoint), vocab=dfjss_nn.VOCAB_REDUCED)}")
    #print(datapoint)
    print(f"Input size: {datapoint.size()}")
    example = datapoint
    break

autoencoder = dfjss_nn.IndividualAutoEncoder()

autoencoder.eval()
autoencoded = autoencoder(example).detach()
print(f"Autoencoded (Untrained): {dfjss_nn.string_from_onehots(autoencoded)}")
#print(autoencoded)
print(f"Output size: {autoencoded.size()}")

print(autoencoder.summary())

dfjss_nn.train_autoencoder(autoencoder,
                           batch_size=128,
                           num_epochs=40,
                           train_size=8192,
                           val_size=1024,
                           regularization_coefficient=10.)
