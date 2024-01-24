import torch
import numpy as np

import time

import dfjss_nn
import dfjss_misc as misc

flatten_trees = True
fill_trees = True

test_set_N = 16384
generated_N = 100000

# test set
dataset = dfjss_nn.AutoencoderDataset(size=test_set_N, flatten_trees=flatten_trees, fill_trees=fill_trees, rng_seed=55338)

# one example is needed to get its reduced form
example = None
for datapoint in dataset:
    print(f"Original: {dfjss_nn.string_from_onehots(datapoint)}")
    print(f"Reduced: {dfjss_nn.string_from_onehots(dfjss_nn.reduce_sequence(datapoint), vocab=dfjss_nn.VOCAB_REDUCED)}")
    #print(datapoint)
    print(f"Input size: {datapoint.size()}")
    example = datapoint
    break

example_reduced = dfjss_nn.reduce_sequence(example)
example_reduced_target = torch.argmax(example_reduced, dim=-1)

state_path = r"Autoencoder input-200-100-200-output\model_epoch297.pth"

autoencoder = dfjss_nn.IndividualFeedForwardAutoEncoder(sequence_length=dataset.max_sequence_length(),
                                                        encoding_size=100,
                                                        encoder_layers_widths=(200,),
                                                        decoder_layers_widths=(200,),
                                                        dropout=0.)

autoencoder.import_state(state_path)

# compute reduced criterion of dataset
print(f"Test set size: {test_set_N}")
print("Computing criterions on the test dataset...", end="")
test_start = time.time()
criterion = torch.nn.NLLLoss()

losses_raw = []
losses_reduced = []
autoencoder.eval()

for i, datapoint in enumerate(dataset):
    datapoint_reduced = dfjss_nn.reduce_sequence(datapoint)

    output = autoencoder(datapoint.unsqueeze(0)).squeeze(0)
    output_reduced = dfjss_nn.reduce_sequence(output, input_is_logs=True)

    target = torch.argmax(datapoint, dim=-1)
    target_reduced = torch.argmax(dfjss_nn.reduce_sequence(datapoint), dim=-1)

    losses_raw.append(criterion(output, target))
    losses_reduced.append(criterion(output_reduced, target_reduced))

    print(f"\rComputing criterions on the test dataset... {misc.timeformat(misc.timeleft(test_start, time.time(), i + 1, test_set_N))}", end="")

losses_raw = torch.stack(losses_raw)
losses_reduced = torch.stack(losses_reduced)

print(f"\nAverage Criterion: {losses_raw.mean().item():.7f} (Raw) {losses_reduced.mean().item():.7f} (Reduced)")

# reduced criterion of decodes of random encodes
print(f"Random encodes amount: {generated_N}")

print("Computing criterions of \"generated\" individuals...", end="")
gen_start = time.time()

rng = np.random.default_rng(7245)

unique_generations = set()

losses_generated = []

for m in range(1, generated_N + 1):
    random_encode = torch.tensor(rng.uniform(low=-1, high=1, size=autoencoder.encoding_size), device=dfjss_nn.device).float()

    decoded = autoencoder.decoder(random_encode.unsqueeze(0)).squeeze(0)
    decoded_reduced = dfjss_nn.reduce_sequence(decoded, input_is_logs=True)

    unique_generations.add(dfjss_nn.string_from_onehots(decoded))
    losses_generated.append(criterion(decoded_reduced, example_reduced_target))

    print(f"\rComputing criterions of \"generated\" individuals... {misc.timeformat(misc.timeleft(gen_start, time.time(), m, generated_N))}", end="")

losses_generated = torch.stack(losses_generated)

print(f"\nAverage Criterion: N/A (Raw) {losses_generated.mean().item():.7f} (Reduced, Generated) {len(unique_generations)} unique out of {generated_N}")

# input("Return to exit")
