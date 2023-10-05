import numpy as np
import torch
import dfjss_nn
import dfjss_genetic
import dfjss_misc as misc
import dfjss_priorityfunction as pf
import time
import matplotlib.pyplot as plt

N_random = 10000
N_valid = 10000
quantiles = [0, 0.25, 0.5, 0.75, 1]

random_strings = []
valid_strings = []
scores_of_random = np.zeros(shape=N_random)
scores_of_valid = np.zeros(shape=N_valid)
true_of_random = np.zeros(shape=N_random, dtype=np.byte)
true_of_valid = np.zeros(shape=N_valid, dtype=np.byte)

rng = np.random.default_rng(seed=100)

distortion = 0.01
distortion_degree = 3
replacement = 0.05

print(f"Distortion = {distortion:.0%} ^ {distortion_degree}, Replacement = {replacement:.0%}")

placeholder_features = ["F"]
placeholder_ops = {"o": pf.pf_add}


def is_valid(r):
    return int(pf.is_representation_valid(r, features=placeholder_features, operations=placeholder_ops))


def do_replacement(sequence):
    if replacement <= 0.:
        return sequence

    for i in range(sequence.shape[0]):
        if rng.uniform() >= replacement:
            continue

        # choose a non-maximum index in the one-hot representation and swap it with the max
        max_index = torch.argmax(sequence[i, :]).item()
        other_index = max_index
        while other_index == max_index:
            other_index = rng.choice(a=sequence.shape[1])

        temp = sequence[i, max_index].item()
        sequence[i, max_index] = sequence[i, other_index]
        sequence[i, other_index] = temp

    return sequence


def disperse_sequence(sequence):
    if distortion <= 0.:
        return sequence

    noises = torch.empty(size=(max(distortion_degree, 1), sequence.shape[0], sequence.shape[1])).uniform_()
    noises = noises / torch.sum(noises, dim=2).unsqueeze(2)

    noise = noises.mean(dim=0)

    return distortion * noise + (1. - distortion) * sequence

start_random = time.time()

# random
for n in range(N_random):
    l = rng.geometric(p=0.25) * 4 + 1
    s = "".join(rng.choice(a=dfjss_nn.VOCAB_REDUCED, size=l))
    random_strings.append(s)
    random_sequence = disperse_sequence(dfjss_nn.one_hot_sequence(s, vocab=dfjss_nn.VOCAB_REDUCED))

    scores_of_random[n] = dfjss_nn.syntax_score(random_sequence)
    true_of_random[n] = is_valid(dfjss_nn.string_from_onehots(random_sequence, vocab=dfjss_nn.VOCAB_REDUCED))

    print(f"\r1/2 {misc.timeformat(misc.timeleft(start_random, time.time(), n+1, N_random))}", end="", flush=True)

print("")


def argquantile(x, q):
    Q = min(int(len(x) * q), len(x) - 1)
    a = np.partition(x, Q)
    return np.argpartition(x, Q)[Q]


for q in quantiles:
    idx = argquantile(scores_of_random, q)
    print(f"Random individuals' {q * 100:.0f}th percentile score: {scores_of_random[idx]:.5f}, {random_strings[idx]}")

# valid
gen_algo = dfjss_genetic.GeneticAlgorithm(rng_seed=100)
gen_algo.settings.features = dfjss_nn.INDIVIDUALS_FEATURES

start_valid = time.time()

for n in range(N_valid):
    representation = repr(gen_algo.get_random_individual())
    reduced = dfjss_nn.string_from_onehots(dfjss_nn.reduce_sequence(dfjss_nn.one_hot_sequence(representation)), vocab=dfjss_nn.VOCAB_REDUCED)

    seq = disperse_sequence(do_replacement(dfjss_nn.reduce_sequence(dfjss_nn.one_hot_sequence(representation))))

    reconstructed = dfjss_nn.string_from_onehots(seq, vocab=dfjss_nn.VOCAB_REDUCED)

    scores_of_valid[n] = dfjss_nn.syntax_score(seq)
    true_of_valid[n] = is_valid(reconstructed)
    valid_strings.append(reconstructed)

    print(f"\r2/2 {misc.timeformat(misc.timeleft(start_valid, time.time(), n + 1, N_valid))}", end="", flush=True)

print(f"\nTook {misc.timeformat(time.time() - start_random)}")

for q in quantiles:
    idx = argquantile(scores_of_valid, q)
    print(f"Valid individuals' {q * 100:.0f}th percentile score: {scores_of_valid[idx]:.5f}, {valid_strings[idx]}")

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex="all", figsize=(15, 25))

ax1.hist(scores_of_random, density=True, alpha=0.5, label="Syntax score of random sequences")
ax1.hist(scores_of_valid, density=True, alpha=0.5, label="Syntax score of valid sequences")

ax1.set_ylabel("Density")

ax2.scatter(scores_of_random, true_of_random, alpha=np.sqrt(1./N_random), label="Random", marker='X', edgecolors='none')
ax2.scatter(scores_of_valid, true_of_valid, alpha=np.sqrt(1./N_random), label="Valid", marker='P', edgecolors='none')

ax1.set_title(f"Syntax score of random vs valid")
ax2.set_title("Syntax score vs True value")

ax2.set_xlabel("Syntax score")
ax2.set_ylabel("True value")

ax1.legend()

plt.suptitle(f"(Distortion={distortion:.0%}^{distortion_degree}, Replacement={replacement:.0%})")

plt.show()
