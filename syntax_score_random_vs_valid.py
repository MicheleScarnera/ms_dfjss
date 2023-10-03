import numpy as np
import dfjss_nn
import dfjss_genetic
import matplotlib.pyplot as plt

N = 2000
scores_of_random = np.zeros(shape=N)
scores_of_valid = np.zeros(shape=N)

rng = np.random.default_rng(seed=100)

# random
for n in range(N):
    l = rng.geometric(p=0.25) * 4 + 1
    random_sequence = dfjss_nn.one_hot_sequence("".join(rng.choice(a=dfjss_nn.VOCAB_REDUCED, size=l)), vocab=dfjss_nn.VOCAB_REDUCED)

    scores_of_random[n] = dfjss_nn.syntax_score(random_sequence)

# valid
gen_algo = dfjss_genetic.GeneticAlgorithm(rng_seed=100)
gen_algo.settings.features = dfjss_nn.INDIVIDUALS_FEATURES

for n in range(N):
    representation = repr(gen_algo.get_random_individual())
    reduced = dfjss_nn.string_from_onehots(dfjss_nn.reduce_sequence(dfjss_nn.one_hot_sequence(representation)), vocab=dfjss_nn.VOCAB_REDUCED)

    scores_of_valid[n] = dfjss_nn.syntax_score(dfjss_nn.one_hot_sequence(representation))

fig, ax1 = plt.subplots(nrows=1, sharex="all")

ax1.hist(scores_of_random, density=True, alpha=0.5, label="Syntax score of random sequences")
ax1.hist(scores_of_valid, density=True, alpha=0.5, label="Syntax score of valid sequences")

#plt.xlim([0, 1])

plt.legend()

plt.show()
