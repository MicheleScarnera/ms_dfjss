import dfjss_nn

dataset = dfjss_nn.IndividualDataset()

for datapoint in dataset:
    print(f"Original: {dataset.df.loc[0, 'Individual']}")
    print(f"Reconstructed: {dfjss_nn.string_from_onehots(datapoint)}")
    #print(datapoint)
    print(f"Input size: {datapoint.size()}")
    example = datapoint
    break

autoencoder = dfjss_nn.IndividualAutoEncoder()

autoencoder.eval()
autoencoded = autoencoder(example).detach()
#print(dfjss_nn.string_from_onehots(autoencoded))
#print(autoencoded)
print(f"Output size: {autoencoded.size()}")

dfjss_nn.train_autoencoder(autoencoder, dataset, batch_size=4, num_epochs=10)
