import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import pandas as pd

import time
import datetime
import os
import warnings

from collections import Counter, defaultdict

import dfjss_objects as dfjss
import dfjss_priorityfunction as pf
import dfjss_genetic as genetic
import dfjss_misc as misc
import dfjss_phenotype as pht

torch.autograd.set_detect_anomaly(True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device.upper()}")

torch.set_default_device(device)

AUTOENCODER_PRECOMPUTE_TABLE_FILENAME = "precompute_table.csv"

REWARDMODEL_FILENAME = "reward_model_dataset.csv"
INDIVIDUALS_FEATURES = ["operation_work_required",
                        "operation_windup",
                        "operation_cooldown",
                        "job_relative_deadline",
                        "job_time_alive",
                        "job_remaining_number_of_operations",
                        "job_remaining_work_to_complete",
                        "job_earliness_penalty",
                        "job_lateness_penalty",
                        "job_delivery_relaxation",
                        "machine_capacity",
                        "machine_cooldown",
                        "machine_current_breakdown_rate",
                        "machine_replacement_cooldown",
                        "warehouse_utilization_rate",
                        "pair_number_of_alternative_machines",
                        "pair_number_of_alternative_operations",
                        "pair_expected_work_power",
                        "pair_expected_processing_time"]

gp_settings = genetic.GeneticAlgorithmSettings()

OPERATIONS = ["+", "-", "*", "/", "<", ">"]
CONSTANTS = [misc.constant_format(num) for num in gp_settings.random_numbers_set()]

FEATURES_AND_CONSTANTS = INDIVIDUALS_FEATURES + CONSTANTS

VOCAB = ["NULL", "EOS", "(", ")"]  # , *OPERATIONS, *INDIVIDUALS_FEATURES, *CONSTANTS
VOCAB_REDUCED = VOCAB.copy()

EOS_INDEX = VOCAB.index("EOS")

FIRST_NONFUNCTIONAL_REDUCED_TOKEN_INDEX = len(VOCAB_REDUCED)

VOCAB_OPERATIONS_LOCATION = (len(VOCAB), len(VOCAB) + len(OPERATIONS))
VOCAB.extend(OPERATIONS)
VOCAB_REDUCED.append("o")

VOCAB_FEATURES_LOCATION = (len(VOCAB), len(VOCAB) + len(INDIVIDUALS_FEATURES) + len(CONSTANTS))
VOCAB.extend(INDIVIDUALS_FEATURES)
VOCAB.extend(CONSTANTS)
VOCAB_REDUCED.append("F")

VOCAB_SIZE = len(VOCAB)
VOCAB_REDUCED_SIZE = len(VOCAB_REDUCED)

VOCAB_REDUCTION_MATRIX = torch.zeros(size=(VOCAB_REDUCED_SIZE, VOCAB_SIZE))
for i in (0, 1, 2, 3):
    VOCAB_REDUCTION_MATRIX[i, i] = 1.

VOCAB_REDUCTION_MATRIX[4, VOCAB_OPERATIONS_LOCATION[0]:VOCAB_OPERATIONS_LOCATION[1]] = 1.
VOCAB_REDUCTION_MATRIX[5, VOCAB_FEATURES_LOCATION[0]:VOCAB_FEATURES_LOCATION[1]] = 1.


def new_first_decoder_token(confidence=1, dtype=torch.float32):
    result = np.full(shape=VOCAB_SIZE, fill_value=0)
    result[VOCAB.index("(")] = confidence
    return torch.tensor(result, dtype=dtype)


# hidden_size=2400, encoding_size=1200

class IndividualFeedForwardAutoEncoder(nn.Module):
    def __init__(self, sequence_length, embedding_dim=-1, encoder_layers_widths=(200,), encoding_size=100,
                 decoder_layers_widths=(200,), dropout=0., enable_batch_norm=True):
        super(IndividualFeedForwardAutoEncoder, self).__init__()

        if embedding_dim > 0:
            self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE,
                                          embedding_dim=embedding_dim,
                                          device=device)
        else:
            self.embedding = None

        self.sequence_length = sequence_length
        self.encoding_size = encoding_size
        self.flat_in_size = sequence_length * (embedding_dim if embedding_dim > 0 else VOCAB_SIZE)
        self.flat_out_size = sequence_length * VOCAB_SIZE

        self.register_buffer("enable_batch_norm", torch.tensor(enable_batch_norm))

        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.encoder_activations = nn.ModuleList()

        self.decoder_layers = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        self.decoder_activations = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

        self.input_flatten = nn.Flatten(-2, -1)

        self.decode_unflatten = nn.Unflatten(dim=-1, unflattened_size=(sequence_length, VOCAB_SIZE))
        self.output_activation = nn.LogSoftmax(dim=-1)

        for is_encoder in (True, False):
            if is_encoder:
                first_width = self.flat_in_size
                last_width = self.encoding_size
                widths = (*encoder_layers_widths, None)
                module_list = self.encoder_layers
                norm_list = self.encoder_norms
                activ_list = self.encoder_activations
            else:
                first_width = self.encoding_size
                last_width = self.flat_out_size
                widths = (*decoder_layers_widths, None)
                module_list = self.decoder_layers
                norm_list = self.decoder_norms
                activ_list = self.decoder_activations

            last_i = len(widths) - 1
            for i, w in enumerate(widths):
                f_in = widths[i - 1] if i > 0 else first_width
                f_out = widths[i] if i < last_i else last_width

                module_list.append(nn.Linear(in_features=f_in, out_features=f_out))
                norm_list.append(nn.BatchNorm1d(f_out))

                if is_encoder and i == last_i:
                    activ_list.append(nn.Tanh())
                else:
                    activ_list.append(nn.PReLU())

    def import_state(self, path):

        self.load_state_dict(torch.load(path))

        self.sequence_length = self.decode_unflatten.unflattened_size[0]
        self.encoding_size = self.encoder_layers[-1].out_features
        self.flat_in_size = self.sequence_length * VOCAB_SIZE  # (embedding_dim if embedding_dim > 0 else VOCAB_SIZE)
        self.flat_out_size = self.sequence_length * VOCAB_SIZE

        self.eval()
        print("Note: autoencoder has been set to eval mode")

    def make_precompute_table(self, individuals, path, batch_size=64, verbose=1):
        self.eval()

        data = {"Individual": [], "Encode": [], "AntiDecode": []}

        start = time.time()
        if verbose > 0:
            print(f"Making pre-compute table for autoencoder...", end="", flush=True)

        i = 1
        I = len(individuals)
        running_batch = []
        for ind in individuals:
            ind = end_with_eos(ind)
            seq = one_hot_sequence(ind)

            data["Individual"].append(ind)

            is_last = i == I

            if len(running_batch) < batch_size:
                running_batch.append(seq)

            if len(running_batch) == batch_size or is_last:
                batch_tensor = torch.stack(running_batch)

                with torch.no_grad():
                    encode = self.encoder(batch_tensor)

                def format_it(tensor):
                    return np.array2string(tensor.numpy(), threshold=self.encoding_size + 1)

                data["Encode"].extend([format_it(e) for e in encode])
                data["AntiDecode"].extend(
                    [format_it(ad) for ad in self.anti_decoder(one_hot_to_sparse(batch_tensor), start_encode=encode)])

                running_batch = []

            if verbose > 0:
                print(
                    f"\rMaking pre-compute table for autoencoder... {(i / I):.2%} ETA {misc.timeformat(misc.timeleft(start, time.time(), i, I))}",
                    end="", flush=True)

            i += 1
        if verbose > 0:
            print(f"\rMade pre-compute table in {misc.timeformat(time.time() - start)}", flush=True)

        df = pd.DataFrame(data)

        if os.path.exists(path):
            df_old = pd.read_csv(path)

            df = pd.concat([df, df_old], ignore_index=True)
            df.drop_duplicates(subset="Individual", inplace=True)

        df.to_csv(path_or_buf=path, index=False, mode="w")

        return df

    def get_precompute_table(self, individuals, path, anti_decode=True, verbose=1):
        if verbose > 0:
            print("Getting/Making dataset's pre-compute table...")

        make_table = False
        df = None
        if os.path.exists(path):
            df = pd.read_csv(path)

            arg_individuals = set([end_with_eos(seq) for seq in individuals])
            table_individuals = set(df["Individual"])

            for arg_ind in arg_individuals:
                if arg_ind not in table_individuals:
                    make_table = True
                    break

            # if not arg_individuals <= table_individuals:
                # make_table = True
        else:
            make_table = True

        if make_table:
            df = self.make_precompute_table(individuals, path, verbose=verbose)

        if type(df.loc[0, "Encode"]) == str or type(df.loc[0, "AntiDecode"]) == str:
            for c in ("Encode", "AntiDecode"):
                df[c] = [np.fromstring(row[c].strip("[]"), dtype=np.float32, sep=" ") for _, row in df.iterrows()]

        C = "AntiDecode" if anti_decode else "Encode"

        return dict([(
            end_with_eos(row["Individual"]),
            torch.tensor(row[C])
        )
            for _, row in df.iterrows()])

    def auto_encoder(self, sequence):
        return self.decoder(self.encoder(sequence))

    def encoder(self, sequence):
        if torch.is_floating_point(sequence):
            if sequence.dim() == 2:
                return self.encoder(sequence.unsqueeze(0)).squeeze(0)

            indices = torch.argmax(sequence, dim=-1)
        else:
            indices = sequence

        embed = self.embedding(indices) if self.embedding is not None else sequence
        result = self.input_flatten(embed)

        last_i = len(self.encoder_layers) - 1
        for i, (layer, norm, activ) in enumerate(
                zip(self.encoder_layers, self.encoder_norms, self.encoder_activations)):
            result = layer(result)

            if self.enable_batch_norm:
                result = norm(result)

            result = activ(result)

            # final encode does not get dropout
            if i < last_i:
                result = self.dropout(result)

        return result

    def decoder(self, encoding):
        if encoding.dim() == 1:
            return self.decoder(encoding.unsqueeze(0)).squeeze(0)

        result = encoding

        last_i = len(self.decoder_layers) - 1
        for i, (layer, norm, activ) in enumerate(zip(self.decoder_layers, self.decoder_norms, self.decoder_activations)):
            result = layer(result)

            if self.enable_batch_norm:
                result = norm(result)

            result = activ(result)

            # final decode does not get dropout
            if i < last_i:
                result = self.dropout(result)

        result = self.output_activation(self.decode_unflatten(result))

        return result

    def anti_decoder(self, desired_output, start_encode=None, gradient_max_norm=0.1, max_iters=100, verbose=0,
                     return_iterations=False):
        if desired_output.dim() > 1 and ((start_encode is not None and start_encode.dim() > 1) or start_encode is None):
            return torch.stack([self.anti_decoder(desired_output[i],
                                                  start_encode[i] if start_encode is not None else None,
                                                  gradient_max_norm=gradient_max_norm,
                                                  max_iters=max_iters,
                                                  verbose=verbose,
                                                  return_iterations=False) for i in range(desired_output.shape[0])])

        if start_encode is not None:
            current_encode = start_encode.detach().requires_grad_()
        else:
            current_encode = self.encoder(sparse_to_one_hot(desired_output)).detach().requires_grad_()

        current_encode.retain_grad()

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(params=[current_encode], lr=0.001)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1. / (1. + 0.01 * epoch))

        iterations = 0
        while True and ((max_iters >= 0 and iterations < max_iters) or max_iters < 0):
            optimizer.zero_grad()

            current_output_logits = self.decoder(current_encode)
            current_output = current_output_logits.argmax(dim=-1)

            if verbose > 0:
                print(
                    f"Iteration {iterations}: \n\tCurrent = {string_from_sparse(current_output)}\n\tDesired = {string_from_sparse(desired_output)}")

            if torch.equal(current_output, desired_output):
                break

            loss = criterion(current_output_logits, desired_output)
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(current_encode,
                                                 max_norm=gradient_max_norm,
                                                 error_if_nonfinite=True)

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                current_encode.clamp_(min=-1, max=1)

            iterations += 1

        if return_iterations:
            return current_encode.detach(), iterations
        else:
            return current_encode.detach()

    def forward(self, x):
        return self.auto_encoder(x)

    def count_parameters(self, learnable=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad == learnable)

    def summary(self):
        longdash = '------------------------------------'
        result = [longdash, "Individual Feed-Forward Auto-Encoder",
                  f"Embedding size: {self.embedding.embedding_dim if self.embedding is not None else 'N/A (using one-hot)'}",
                  f"Max Length: {self.sequence_length}",
                  f"Flat size: {self.flat_in_size} (In), {self.flat_out_size} (Out)",
                  f"Encoder layers: {len(self.encoder_layers)} ({', '.join([str(layer.in_features) + '->' + str(layer.out_features) for layer in self.encoder_layers])})",
                  f"Encoding Size: {self.encoding_size}",
                  f"Decoder layers: {len(self.decoder_layers)} ({', '.join([str(layer.in_features) + '->' + str(layer.out_features) for layer in self.decoder_layers])})",
                  f"Dropout: {self.dropout.p}",
                  f"Batch Normalization: {'Yes' if self.enable_batch_norm else 'No'}",
                  f"Number of parameters: {self.count_parameters()} (Learnable), {self.count_parameters(False)} (Fixed)",
                  longdash]

        return "\n".join(result)


def one_hot_sequence(individual, vocab=None):
    if type(individual) == str:
        S = tokenize_with_vocab(individual, vocab=vocab)
    elif type(individual) == list and type(individual[0]) == str:
        S = individual
    else:
        raise ValueError(f"individual neither a str or list of str ({type(individual)})")

    return torch.stack([token_to_one_hot(s, vocab=vocab) for s in S])


def sparse_sequence(individual, vocab=None):
    return torch.stack([token_to_sparse(s, vocab=vocab) for s in tokenize_with_vocab(individual, vocab=vocab)])


def one_hot_to_sparse(sequence):
    return sequence.argmax(dim=-1)


def sparse_to_one_hot(sequence, vocab=None):
    if vocab is None:
        vocab = VOCAB

    return torch.nn.functional.one_hot(sequence, num_classes=len(vocab)).to(device, torch.float32)


def reduce_sequence(sequence, input_is_logs=False):
    if len(sequence.shape) == 3:
        return torch.stack([reduce_sequence(x_, input_is_logs) for x_ in sequence], dim=0)

    if sequence.shape[1] == VOCAB_REDUCED_SIZE:
        raise ValueError("Sequence is already reduced")

    if sequence.shape[1] != VOCAB_SIZE:
        raise ValueError(
            f"Sequence is of unexpected vocabulary (expected vocabulary of size {VOCAB_SIZE}, got {sequence.shape[1]})")

    if input_is_logs:
        return torch.log(torch.softmax(sequence, dim=1) @ VOCAB_REDUCTION_MATRIX.T)
    else:
        return sequence @ VOCAB_REDUCTION_MATRIX.T


def token_to_one_hot(s, vocab=None):
    if vocab is None:
        vocab = VOCAB

    # Convert a token to a one-hot tensor
    one_hot = torch.zeros(len(vocab))
    if s in vocab:
        index = vocab.index(s)
        one_hot[index] = 1
    else:
        return token_to_one_hot("NULL")
    return one_hot


def token_to_sparse(s, vocab=None):
    if vocab is None:
        vocab = VOCAB

    # Convert a token to a one-hot tensor
    one_hot = torch.zeros(len(vocab))
    if s in vocab:
        index = vocab.index(s)
    else:
        index = vocab.index("NULL")
    return torch.tensor(index)


def tokenize_with_vocab(input_string, vocab=None):
    if vocab is None:
        vocab = VOCAB

    # Sort the vocab list by length in descending order
    sorted_vocab = sorted(vocab, key=lambda x: len(x), reverse=True)

    tokens = []
    i = 0

    while i < len(input_string):
        matched = False

        # possible_words = []
        for word in sorted_vocab:
            if input_string[i:i + len(word)] == word:
                # possible_words.append(word)
                tokens.append(word)
                i += len(word)
                matched = True
                break

        if not matched:
            raise ValueError(f"Could not match character '{input_string[i]}' with anything in the vocabulary {vocab}")

    return tokens


def string_from_onehots(onehots, vocab=None, list_mode=False):
    if len(onehots.shape) > 3:
        raise ValueError(f"Too many dimensions ({len(onehots.shape)})")

    if len(onehots.shape) == 3:
        return [string_from_onehots(onehots[i, :], vocab) for i in range(onehots.shape[0])]

    if len(onehots.shape) != 2:
        raise ValueError(f"onehots is not 2-D, but {len(onehots.shape)}-D {onehots.shape}")

    if vocab is None:
        vocab = VOCAB

    result = [] if list_mode else ""
    for onehot in onehots:
        if list_mode:
            result.append(vocab[torch.argmax(onehot).item()])
        else:
            result += vocab[torch.argmax(onehot).item()]

    return result


def string_from_sparse(sparses, vocab=None, list_mode=False):
    if len(sparses.shape) > 2:
        raise ValueError(f"Too many dimensions ({len(sparses.shape)})")

    if len(sparses.shape) == 2:
        return [string_from_onehots(sparses[i, :], vocab) for i in range(sparses.shape[0])]

    if vocab is None:
        vocab = VOCAB

    result = [] if list_mode else ""
    for index in sparses:
        if list_mode:
            result.append(vocab[index])
        else:
            result += vocab[index]

    return result


def all_features_are_constants(sequence):
    if type(sequence) == str:
        T = tokenize_with_vocab(sequence)
    elif type(sequence[0]) == str:
        T = sequence
    else:
        T = string_from_onehots(sequence, list_mode=True)

    return not np.any([(token in INDIVIDUALS_FEATURES) for token in T])


# "house": sequence of tokens "(FoF)"

HOUSE_DETECTOR_COORDINATES = [(0, 2), (1, 5), (2, 4), (3, 5), (4, 3)]

HOUSE_DETECTOR_FILTER = torch.zeros(size=(5, VOCAB_REDUCED_SIZE))

for i, j in HOUSE_DETECTOR_COORDINATES:
    HOUSE_DETECTOR_FILTER[i, j] = 1.

HOUSE_DETECTOR_MASK = torch.gt(HOUSE_DETECTOR_FILTER, 0.)


def gmean(vector, epsilon=0.1):
    return (vector + epsilon).prod() ** (1. / vector.shape[0]) - epsilon


def detect_houses(x):
    if len(x.shape) > 2:
        raise ValueError(f"Too many dimensions ({len(x.shape)})")

    if x.shape[1] == VOCAB_SIZE:
        y = x @ VOCAB_REDUCTION_MATRIX.T
        # raise ValueError(f"Sequence must be in reduced form (Vocab of size {VOCAB_REDUCED_SIZE}), while vocab is of size {x.shape[1]}")
    else:
        y = x

    if y.shape[0] % 4 != 1:
        seq_length = y.shape[0] - y.shape[0] % 4 + 1
        # raise ValueError(f"Sequence length must be 1 mod 4, got {sequence.shape[0]} ({sequence.shape[0] % 4} mod 4) instead")
    else:
        seq_length = y.shape[0]

    sequence = y[0:seq_length, :]

    l = []

    for i in range(0, sequence.shape[0] - 4):
        window = sequence[i:(i + 5), :]
        v = torch.masked_select(window, HOUSE_DETECTOR_MASK)

        l.append(gmean(v))

    return torch.stack(l)


def syntax_score(x, aggregate_with_gmean=True):
    if len(x.shape) > 3:
        raise ValueError(f"Too many dimensions ({len(x.shape)})")

    if len(x.shape) == 3:
        return torch.cat([syntax_score(x[i, :], aggregate_with_gmean) for i in range(x.shape[0])])

    if x.shape[1] == VOCAB_SIZE:
        y = x @ VOCAB_REDUCTION_MATRIX.T
    else:
        y = x

    if y.shape[0] % 4 != 1:
        seq_length = y.shape[0] - y.shape[0] % 4 + 1
    else:
        seq_length = y.shape[0]

    sequence = y[0:seq_length, :]

    running_result = None
    # reconstruction = string_from_onehots(sequence, vocab=VOCAB_REDUCED)
    chosen_scores = []
    i = 0
    while running_result is None or running_result.shape[0] > 1:
        # reconstruction = string_from_onehots(sequence, vocab=VOCAB_REDUCED)
        running_result = detect_houses(sequence)

        most_housey_idx = torch.argmax(running_result).item()
        chosen_scores.append(running_result[most_housey_idx])

        mask_before = torch.tensor(
            [[r < most_housey_idx for c in range(VOCAB_REDUCED_SIZE)] for r in range(seq_length)], dtype=torch.bool)
        mask_between = torch.tensor([[most_housey_idx <= r <= most_housey_idx + 4 for c in range(VOCAB_REDUCED_SIZE)]
                                     for r in range(seq_length)], dtype=torch.bool)
        mask_after = torch.tensor(
            [[r > most_housey_idx + 4 for c in range(VOCAB_REDUCED_SIZE)] for r in range(seq_length)], dtype=torch.bool)

        crammed = torch.zeros(size=[VOCAB_REDUCED_SIZE]).scatter_(0, torch.tensor([5]), running_result[
            most_housey_idx]).unsqueeze_(
            0)  # torch.masked_select(sequence, mask_between).view((5, VOCAB_REDUCED_SIZE)) * HOUSE_DETECTOR_FILTER

        pile = []

        if most_housey_idx > 0:
            pile.append(torch.masked_select(sequence, mask_before).view((most_housey_idx, VOCAB_REDUCED_SIZE)))

        pile.append(crammed)

        if most_housey_idx < seq_length - 4:
            pile.append(torch.masked_select(sequence, mask_after).view(
                (seq_length - (most_housey_idx + 5), VOCAB_REDUCED_SIZE)))

        sequence = torch.cat(pile, dim=0)
        seq_length = sequence.shape[0]

        i += 1

    return gmean(torch.stack(chosen_scores, dim=0)) if aggregate_with_gmean else torch.stack(chosen_scores,
                                                                                             dim=0).mean()


class AutoencoderDataset(data.Dataset):
    def __init__(self, rng_seed=100, max_depth=4, size=5000,
                 refresh_rate=1., refresh_is_random=False,
                 inflate=False, inflate_is_multiplicative=False, inflate_max_size=None,
                 fill_trees=True, flatten_trees=False, features_weight_in_full_trees=None,
                 sparse=False):
        """
        Creates a dataset for the autoencoders, made of randomly generated individuals. It has options to "refresh", replacing a proportion of the datapoints.

        :param rng_seed: The seed of the dataset.
        :param max_depth: The maximum depth of individuals.
        :param size: The size of the dataset.
        :param refresh_rate: On a refresh, the proportion of datapoints that is replaced.
        :param inflate: If True, old datapoints are not removed, and the dataset grows in size at every refresh.
        :param inflate_is_multiplicative: If True, inflation is multiplicative. If False, inflation is additive.
        :param inflate_max_size: If not None, the dataset won't grow past this amount.
        :param refresh_is_random: If True, the replaced datapoints are chosen randomly, and the new ones are placed randomly. If False, it's from the beginning and end of the dataset, respectively.
        :param fill_trees: If True, the individuals are always full trees.
        :param flatten_trees: If True, individuals are "flattened", getting rid of parentheses. It is not recommended to flatten trees if they are NOT full.
        :param features_weight_in_full_trees: If not None, it makes features X times more likely than constants, if fill_trees is True. If None, they are not particularly weighted.
        :param sparse: If True, the datapoints are sparse encodings rather than one-hot.
        """
        super().__init__()

        self.gen_algo = genetic.GeneticAlgorithm(rng_seed=rng_seed)
        self.gen_algo.settings.features = INDIVIDUALS_FEATURES
        self.gen_algo.settings.tree_generation_max_depth = max_depth
        self.gen_algo.settings.tree_generation_fill = fill_trees

        self.max_depth = max_depth
        self.fill_trees = fill_trees
        self.flatten_trees = flatten_trees
        self.features_weight_in_full_trees = features_weight_in_full_trees
        self.size = size
        self.initial_size = size
        self.refresh_rate = refresh_rate
        self.inflate = inflate
        self.inflate_is_multiplicative = inflate_is_multiplicative
        self.inflate_max_size = inflate_max_size
        self.refresh_is_random = refresh_is_random
        self.sparse = sparse

        self.individuals = []
        self.times_refreshed = 0
        self.total_datapoints = 0

        self.rng = np.random.default_rng(rng_seed)

        self.generate_individuals()

    def max_sequence_length(self):
        if self.flatten_trees:
            return 2 ** (self.max_depth + 1) - 1 + 1  # no. of parentheses: 2 ** (self.max_depth + 1) - 2. +1 is EOS
        else:
            return 2 ** (self.max_depth + 2) - 2  # without EOS, it would be 2 ** (self.max_depth + 2) - 3

    def random_full_individual(self):
        tree = pf.representation_to_priority_function_tree("({0.0}+{0.0})")

        tree.fill(self.max_depth)

        flat = tree.flatten()

        for i, f in enumerate(flat):
            if f in OPERATIONS:
                flat[i] = self.rng.choice(a=OPERATIONS)
            elif f in FEATURES_AND_CONSTANTS or misc.is_number(f):
                if self.features_weight_in_full_trees is None:
                    flat[i] = self.rng.choice(a=FEATURES_AND_CONSTANTS)
                else:
                    if self.rng.uniform() < 1. / (1. + self.features_weight_in_full_trees):
                        flat[i] = self.rng.choice(a=CONSTANTS)
                    else:
                        flat[i] = self.rng.choice(a=INDIVIDUALS_FEATURES)
            else:
                raise ValueError(f"Unexpected token {f}")

        tree.set_from_flattened(flat)

        return tree

    def generate_individuals(self):
        depths = [d for d in range(2, self.max_depth + 1)]
        while len(self.individuals) < self.size:
            self.gen_algo.settings.tree_generation_max_depth = self.rng.choice(depths)

            individual = None
            while individual is None or individual.depth() != self.max_depth:
                if self.fill_trees:
                    individual = self.random_full_individual()
                else:
                    individual = self.gen_algo.get_random_individual()

                if self.flatten_trees:
                    representation = "".join(individual.flatten()) + "EOS"
                else:
                    representation = repr(individual) + "EOS"

            # del individual

            individual_tensor = sparse_sequence(representation).to(device) if self.sparse else (
                one_hot_sequence(representation).to(device, torch.float32))

            if self.refresh_is_random:
                self.individuals.insert(self.rng.choice(len(self.individuals) + 1), individual_tensor)
            else:
                self.individuals.append(individual_tensor)

            self.total_datapoints += 1

    def refresh_data(self):
        # permanence rate
        if self.refresh_rate > 0.:
            if self.inflate:
                increment = int(
                    (self.size if self.inflate_is_multiplicative else self.initial_size) * self.refresh_rate)

                self.size = self.size + increment if self.inflate_max_size is None else (
                    min(self.size + increment, self.inflate_max_size))
            else:
                for _ in range(int(self.size * self.refresh_rate)):
                    index = self.rng.choice(len(self.individuals)) if self.refresh_is_random else 0

                    self.individuals.pop(index)

        # if length less than size, generate individuals
        self.generate_individuals()

        self.times_refreshed += 1

    def __iter__(self):
        return self.data_iterator()

    def __getitem__(self, idx):
        return self.individuals[idx]

    def __len__(self):
        return len(self.individuals)

    def data_iterator(self):
        for individual in self.individuals:
            yield individual


def num_diagonals(n, include_self_diagonals=False):
    return n * (n + 1) // 2 if include_self_diagonals else n * (n - 1) // 2


def num_diagonals_inverse(x, include_self_diagonals=False, return_int=True):
    b = 1 if include_self_diagonals else -1

    result = 0.5 * (-b + np.sqrt(8 * x + 1))
    return int(result) if return_int else result


def random_unordered_pairs(no_elements, no_pairs=None, torch_rng=None, include_self_pairs=True):
    d = num_diagonals(no_elements, include_self_pairs)

    if no_pairs is None:
        no_pairs = d
    elif no_pairs > d:
        raise ValueError(f"Number of requested pairs ({no_pairs}) is higher than the number of possible pairs ({d})")

    combs = torch.combinations(torch.arange(no_elements), r=2, with_replacement=include_self_pairs)

    perm = torch.randperm(len(combs), generator=torch_rng)

    return combs[perm[:no_pairs]]


class EncoderSimilarityDataset(data.Dataset):
    def __init__(self, autoencoder_dataset, rng_seed=450, number_of_individuals_percent=1., sets_of_features_size=2000,
                 num_of_mutated_per_individual=4, mutated_avg_changes_percent=0.2):
        """
        :type autoencoder_dataset: AutoencoderDataset

        :param autoencoder_dataset:
        :param rng_seed:
        """
        super().__init__()

        number_of_individuals_percent = np.clip(number_of_individuals_percent, 0., 1.)

        self.autoencoder_dataset = autoencoder_dataset

        self.size_from_dataset = int(self.autoencoder_dataset.size * number_of_individuals_percent)
        self.num_of_mutated_per_individual = num_of_mutated_per_individual
        self.size = self.size_from_dataset * self.num_of_mutated_per_individual

        self.mutated_avg_changes_percent = torch.tensor(mutated_avg_changes_percent, dtype=torch.float32)

        self.rng = torch.Generator()
        self.rng.manual_seed(rng_seed)

        self.sets_of_features_size = sets_of_features_size
        self.sets_of_features = self.get_sets_of_features()

        self.original_individuals_data = torch.zeros(
            size=(self.size_from_dataset, *tuple(autoencoder_dataset[0].shape)))
        self.mutated_individuals_data = torch.zeros(size=(self.size, *tuple(autoencoder_dataset[0].shape)))

        self.similarities = torch.zeros(size=(self.size,), dtype=torch.float32)
        self.last_times_refreshed = float("-inf")

        self.total_datapoints = 0

        self.refresh_data()

    def get_sets_of_features(self, from_warehouse=True, seed=123, center_std=3., features_std=1., feature_abs=True):
        if from_warehouse:
            pht_map = pht.PhenotypeMapper(reference_scenarios_amount=self.sets_of_features_size, scenarios_seed=seed)

            return pht_map.scenarios
        else:
            rng = torch.Generator()

            rng.manual_seed(seed)

            centers = dict()

            sets_of_features = []

            for c in INDIVIDUALS_FEATURES:
                centers[c] = torch.normal(mean=0., std=center_std, size=(1,), generator=rng).item()

            for _ in range(self.sets_of_features_size):
                feature_values = dict()

                for c in INDIVIDUALS_FEATURES:
                    feature_values[c] = torch.normal(mean=centers[c], std=features_std, size=(1,), generator=rng)

                    if feature_abs:
                        feature_values[c] = torch.abs(feature_values[c])

                    feature_values[c] = feature_values[c].item()

                sets_of_features.append(feature_values)

            return sets_of_features

    def get_sim_and_values(self, priority_functions, precomputed_priority_values=None):
        """
        :type priority_functions: list[pf.PriorityFunctionTree]
        :type precomputed_priority_values: list[torch.Tensor]

        :param sets_of_features:
        :param precomputed_priority_values:
        :return:
        """
        with torch.no_grad():
            num_functions = len(priority_functions)

            if num_functions <= 1:
                raise ValueError(f"Not enough priority functions. Need at least 2, got {num_functions}")

            k = len(self.sets_of_features)

            priority_values = torch.zeros(size=(num_functions, k), dtype=torch.float32)

            for f in range(num_functions):
                has_precompute = precomputed_priority_values is not None and f < len(precomputed_priority_values)

                if has_precompute:
                    priority_values[f] = precomputed_priority_values[f]
                else:
                    for i in range(k):
                        priority_values[f, i] = priority_functions[f].run(features=self.sets_of_features[i])

            return torch.corrcoef(priority_values).nan_to_num_(), priority_values

    def mutate_priority_function(self, pf_original):
        if not self.autoencoder_dataset.flatten_trees:
            raise NotImplementedError()

        flat = pf_original.flatten()
        num_changes = 0
        while not (0 < num_changes < len(flat)):
            # num_changes = int(torch.poisson(input=self.mutated_avg_changes, generator=self.rng).item())
            num_changes = int(torch.multinomial(
                input=torch.tensor([1. - self.mutated_avg_changes_percent, self.mutated_avg_changes_percent]),
                num_samples=len(flat),
                replacement=True,
                generator=self.rng).sum().item())

        change_at = torch.randperm(len(flat), generator=self.rng)[0:num_changes]

        for j in change_at:
            if flat[j] in FEATURES_AND_CONSTANTS:
                k = FEATURES_AND_CONSTANTS.index(flat[j])
                K = list(range(len(FEATURES_AND_CONSTANTS)))
                K.remove(k)

                flat[j] = FEATURES_AND_CONSTANTS[
                    K[torch.randint(low=0, high=len(FEATURES_AND_CONSTANTS) - 1, size=(1,), generator=self.rng).item()]]
            elif flat[j] in OPERATIONS:
                k = OPERATIONS.index(flat[j])
                K = list(range(len(OPERATIONS)))
                K.remove(k)

                flat[j] = OPERATIONS[
                    K[torch.randint(low=0, high=len(OPERATIONS) - 1, size=(1,),
                                    generator=self.rng).item()]]
            else:
                raise Exception("flat[j] neither in FEATURES_AND_CONSTANTS or OPERATIONS")

        result = pf.representation_to_priority_function_tree("({0.0}+{0.0})", features=INDIVIDUALS_FEATURES)
        result.fill(self.autoencoder_dataset.max_depth)
        result.set_from_flattened(flat)

        return result

    def refresh_data(self):
        # note: this does not refresh the autoencoder dataset itself
        if self.autoencoder_dataset.times_refreshed <= self.last_times_refreshed:
            warnings.warn(
                "EncoderSimilarityDataset.refresh_data() was called more than once on the same refresh of its AutoencoderDataset")

        self.last_times_refreshed = self.autoencoder_dataset.times_refreshed

        indices = torch.randperm(n=len(self.autoencoder_dataset))[0:self.size_from_dataset]

        # make priority functions
        start = time.time()

        for i, ind in enumerate(indices):
            individual_tensor = self.autoencoder_dataset[ind]

            self.original_individuals_data[i] = self.autoencoder_dataset[ind]

            if self.autoencoder_dataset.flatten_trees:
                ind_tokens = string_from_onehots(individual_tensor, list_mode=True)
                pr_func_original = pf.representation_to_priority_function_tree("({0.0}+{0.0})",
                                                                               features=INDIVIDUALS_FEATURES)
                pr_func_original.fill(self.autoencoder_dataset.max_depth)
                pr_func_original.set_from_flattened(ind_tokens)

                pr_func_original_precompute = None
                mutated_pr_funcs = []
                bulk_corr = True

                for j in range(self.num_of_mutated_per_individual):
                    pr_func_mutated = self.mutate_priority_function(pr_func_original)

                    if self.autoencoder_dataset.flatten_trees:
                        mutated_tensor = one_hot_sequence(pr_func_mutated.flatten() + ["EOS"])
                    else:
                        raise NotImplementedError()

                    k = self.num_of_mutated_per_individual * i + j

                    self.mutated_individuals_data[k] = mutated_tensor

                    if bulk_corr:
                        mutated_pr_funcs.append(pr_func_mutated)
                    else:
                        sim, values = self.get_sim_and_values([pr_func_original, pr_func_mutated],
                                                              pr_func_original_precompute)

                        if pr_func_original_precompute is None:
                            pr_func_original_precompute = [values[0]]

                        self.similarities[k] = sim[0, 1]

                if bulk_corr:
                    sim, values = self.get_sim_and_values([pr_func_original] + mutated_pr_funcs)

                    for j in range(self.num_of_mutated_per_individual):
                        k = self.num_of_mutated_per_individual * i + j

                        self.similarities[k] = sim[0, j + 1]
            else:
                raise NotImplementedError()

            print(f"\r{misc.timeformat(misc.timeleft(start, time.time(), i + 1, self.size_from_dataset))}", end="")

        self.total_datapoints += self.size_from_dataset

    def __iter__(self):
        return self.data_iterator()

    def __getitem__(self, idx):
        return torch.stack([self.original_individuals_data[idx // self.num_of_mutated_per_individual],
                            self.mutated_individuals_data[idx]], dim=0), \
               self.similarities[idx]

    def __len__(self):
        return self.size

    def data_iterator(self):
        for i in range(self.size):
            yield self[i]


def individual_sequence_collate(batch):
    longest_sequence_length = max([sample.shape[0] for sample in batch])

    collated_batch = []

    for sample in batch:
        if sample.shape[0] < longest_sequence_length:
            filler = torch.zeros((longest_sequence_length - sample.shape[0], VOCAB_SIZE))
            filler[:, 0] = 1.  # fill with NULL

            sample = torch.cat((sample, filler), dim=0)

        collated_batch.append(sample)

    return torch.stack(collated_batch)


def syntax_penalty_term_and_syntax_score(sequence, lam=10., eps=0.01):
    score = syntax_score(torch.softmax(sequence, dim=1))

    return lam / np.log(eps) * torch.log(score + eps), score


def train_autoencoder(model,
                      num_epochs=10,
                      encoder_only_epochs=2,
                      enable_encoder_specific_training=True,
                      batch_size=16,
                      max_depth=8,
                      flatten_trees=True,
                      fill_trees=True,
                      train_autoencoder_size=5000,
                      train_autoencoder_refresh_rate=1.,
                      train_autoencoder_seed=100,
                      val_autoencoder_size=1000,
                      val_autoencoder_refresh_rate=0.,
                      val_autoencoder_seed=1337,
                      val_autoencoder_perfects_100_patience=5,
                      train_encoder_size_percent=1.,
                      train_encoder_size_mutated=4,
                      val_encoder_size_percent=1.,
                      val_encoder_size_mutated=4,
                      encoder_sets_of_features_size=4000,
                      start_lr_encoder=1e-3,
                      start_lr_decoder=1e-3,
                      raw_criterion_weight=0.3,
                      raw_criterion_weight_inc=0.01,
                      reduced_criterion_weight=0.7,
                      reduced_criterion_weight_inc=0.,
                      feature_classes_weight=4,
                      operation_classes_weight=2,
                      other_classes_weight=1,
                      gradient_value_threshold=1.,
                      clipping_is_norm=True,
                      gradient_norm_threshold=1.,
                      clipping_norm_type=2.,
                      make_examples_csv=False):
    """
    Train an autoencoder.

    :type model: IndividualFeedForwardAutoEncoder

    :param model: The autoencoder model.
    :param max_depth: Individuals will not exceed this depth.
    :param flatten_trees: Whether data should be of flat individuals or not. You cannot have flat individuals if they are not also full.
    :param fill_trees: Whether individuals should be full. You cannot have flat individuals if they are not also full.
    :param num_epochs: The maximum number of epochs to train on.
    :param train_autoencoder_size: The size of the training set each epoch.
    :param train_autoencoder_refresh_rate: How much, in percentage, the training set is replaced with new data each epoch.
    :param train_autoencoder_seed: The seed for the training data.
    :param val_autoencoder_size: The size of the validation set each epoch. If the training set has non-zero refresh rate, it makes sense for this to be large, even larger than train_size.
    :param val_autoencoder_refresh_rate: How much, in percentage, the validation set is replaced with new data each epoch. It is recommended to keep this to 0 for consistency of evaluation.
    :param val_autoencoder_seed: The seed for the validation data.
    :param val_autoencoder_perfects_100_patience: If the 'Perfects' metric in the validation set reaches 100%, training is stopped if it's kept at 100% for 'val_perfects_100_patience' epochs in a row.
    :param raw_criterion_weight: The relative weight of the raw criterion at epoch 1.
    :param raw_criterion_weight_inc: Every epoch, the relative weight of the raw criterion is increased by this amount.
    :param reduced_criterion_weight: The relative weight of the reduced criterion at epoch 1.
    :param reduced_criterion_weight_inc: Every epoch, the relative weight of the reduced criterion is increased by this amount.
    :param feature_classes_weight: In the raw criterion, the relative weight of features.
    :param operation_classes_weight: In the raw criterion, the relative weight of operations.
    :param other_classes_weight: In the raw criterion, the relative weight of characters that are neither features or operations.
    :param batch_size: The size of batches.
    :param syntax_score_coefficient: If >0, syntax score is explicitely optimized. NOT RECOMMENDED
    :param gradient_value_threshold: If gradient clipping is done by value, this is the farther each individual gradient can be from 0.
    :param clipping_is_norm: Whether the clipping is done by norm (True) or by value (False)
    :param gradient_norm_threshold: If gradient clipping is done by norm, this is the largest the norm of the gradients can be.
    :param clipping_norm_type: The type of norm.
    :return:
    """

    if flatten_trees and not fill_trees:
        raise ValueError(
            "You cannot have flat individuals if they are not also full. Change the flatten_trees and fill_trees arguments")

    folder_name = datetime.datetime.now().strftime(f'AUTOENCODER %Y-%m-%d %H-%M-%S')

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print(f"Model(s) will be saved in \"{folder_name}\"")

    with open(f"{folder_name}/model_summary.txt", "w") as model_summary_file:
        model_summary_file.write(model.summary())

    df_cols = ["Epoch", "Criterion_Weight_Raw", "Criterion_Weight_Reduced"]

    for tv in ("Train", "Val"):
        for q in (
                "TotalDatapoints", "Loss", "Total_Criterion", "Raw_Criterion", "Reduced_Criterion", "SyntaxScore",
                "Valid",
                "Accuracy", "Perfects"):
            df_cols.append(f"{tv}_{q}")

        if tv == "Train":
            df_cols.append("Train_LR")

    df_cols.extend(("Example", "AutoencodedExample"))

    df_examples_cols = ["Epoch", "Example", "AutoencodedExample", "Accuracy"]

    df = pd.DataFrame(columns=df_cols)

    df_examples = pd.DataFrame(columns=df_examples_cols)

    def sims_summary(sims, eps=0.01):
        below_minus_eps = (sims < -eps).type(torch.float32).mean().item()
        above_eps = (sims > eps).type(torch.float32).mean().item()
        result = [f"Deciles of similarities:\n{torch.quantile(sims, q=torch.linspace(0., 1., 10)).numpy()}",
                  f"Below {-eps}: {below_minus_eps:.2%}",
                  f"Between {-eps} and {eps}: {1. - below_minus_eps - above_eps:.2%}",
                  f"Above {eps}: {above_eps:.2%}"]

        return "\n".join(result)

    print("Making training dataset... ", end="")
    t = time.time()
    train_autoencoder_set = AutoencoderDataset(max_depth=max_depth,
                                               size=train_autoencoder_size,
                                               refresh_rate=train_autoencoder_refresh_rate,
                                               flatten_trees=flatten_trees,
                                               fill_trees=fill_trees,
                                               rng_seed=train_autoencoder_seed)

    if enable_encoder_specific_training:
        train_encoder_set = EncoderSimilarityDataset(autoencoder_dataset=train_autoencoder_set,
                                                     number_of_individuals_percent=train_encoder_size_percent,
                                                     num_of_mutated_per_individual=train_encoder_size_mutated,
                                                     sets_of_features_size=encoder_sets_of_features_size,
                                                     rng_seed=train_autoencoder_seed)

        print(sims_summary(train_encoder_set.similarities))

    print(f"Took {misc.timeformat(time.time() - t)}")

    print("Making validation dataset... ", end="")
    t = time.time()
    val_autoencoder_set = AutoencoderDataset(max_depth=max_depth,
                                             size=val_autoencoder_size,
                                             refresh_rate=val_autoencoder_refresh_rate,
                                             flatten_trees=flatten_trees,
                                             fill_trees=fill_trees,
                                             rng_seed=val_autoencoder_seed)

    if enable_encoder_specific_training:
        val_encoder_set = EncoderSimilarityDataset(autoencoder_dataset=val_autoencoder_set,
                                                   number_of_individuals_percent=val_encoder_size_percent,
                                                   num_of_mutated_per_individual=val_encoder_size_mutated,
                                                   sets_of_features_size=encoder_sets_of_features_size,
                                                   rng_seed=val_autoencoder_seed)

        print(sims_summary(val_encoder_set.similarities))

    print(f"Took {misc.timeformat(time.time() - t)}")

    implied_length = train_autoencoder_set.max_sequence_length()
    if model.sequence_length != implied_length:
        raise ValueError(
            f"The feed-forward network needs individuals whose length is exactly {model.sequence_length}, but the max_depth ({max_depth}) parameter implies a length of {implied_length}.")

    print(f"Creating training and validation set for Epoch 1...", end="")

    if enable_encoder_specific_training:
        train_encoder_loader = data.DataLoader(
            train_encoder_set,
            batch_size=batch_size,
            shuffle=True
        )

        val_encoder_loader = data.DataLoader(
            val_encoder_set,
            batch_size=batch_size,
            shuffle=True
        )

    train_autoencoder_loader = data.DataLoader(
        train_autoencoder_set,
        batch_size=batch_size,
        collate_fn=individual_sequence_collate,
        shuffle=True
    )

    val_autoencoder_loader = data.DataLoader(
        val_autoencoder_set,
        batch_size=batch_size,
        collate_fn=individual_sequence_collate,
        shuffle=True
    )

    # train_set, val_set = data.random_split(dataset=dataset, lengths=[1. - val_split, val_split])

    def class_weight(token):
        if token in INDIVIDUALS_FEATURES:
            return feature_classes_weight
        elif token in OPERATIONS:
            return operation_classes_weight
        else:
            return other_classes_weight

    encoder_criterion = nn.MSELoss(reduction='none')
    max_dissimilar_loss = encoder_criterion(torch.full(size=(model.encoding_size,), fill_value=-1.),
                                            torch.full(size=(model.encoding_size,), fill_value=1.)).sum()
    peg = 0.
    center = max_dissimilar_loss  # 0.5 * max_dissimilar_loss

    raw_criterion_class_weights = torch.tensor([class_weight(token) for token in VOCAB], dtype=torch.float32)
    autoencoder_criterion_raw = nn.NLLLoss(weight=raw_criterion_class_weights)
    autoencoder_criterion_reduced = nn.NLLLoss()

    encoder_params, decoder_params = [], []
    doing_encoder = True

    for name, param in model.named_parameters():
        if doing_encoder and "decoder" in name:
            doing_encoder = False

        if doing_encoder:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = optim.Adam([{'params': encoder_params, 'lr': start_lr_encoder},
                            {'params': decoder_params, 'lr': start_lr_decoder}])

    threshold, patience, factor = 1e-8, 10, 0.1 ** 0.25

    autoencoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max",
                                                                 threshold=threshold, patience=patience, cooldown=0,
                                                                 factor=factor,
                                                                 verbose=False)

    perfect_epochs_in_a_row = 0

    for epoch in range(1, num_epochs + 1):
        # ENCODER
        train_encoder_loss = 0.

        train_encoder_start = None
        train_encoder_progress = 0
        train_encoder_progress_needed = None

        val_encoder_loss = 0.

        val_encoder_start = None
        val_encoder_progress = 0
        val_encoder_progress_needed = None

        # DECODER

        skip_decoder = (epoch <= encoder_only_epochs) and enable_encoder_specific_training

        # Training
        train_autoencoder_loss = 0.
        train_autoencoder_raw_criterion = 0.
        train_autoencoder_reduced_criterion = 0.
        train_autoencoder_total_criterion = 0.
        train_autoencoder_largest_criterion = 0.

        train_autoencoder_accuracy = 0.
        train_autoencoder_valid = 0.
        train_autoencoder_perfect_matches = 0

        train_autoencoder_start = None
        train_autoencoder_progress = 0
        train_autoencoder_progress_needed = None

        # Validation

        val_autoencoder_loss = 0.
        val_autoencoder_raw_criterion = 0.
        val_autoencoder_reduced_criterion = 0.
        val_autoencoder_total_criterion = 0.

        val_autoencoder_accuracy = 0.
        val_autoencoder_valid = 0.
        val_autoencoder_perfect_matches = 0

        val_autoencoder_example = ""
        val_autoencoder_autoencoded = ""

        val_autoencoder_examples = []

        val_autoencoder_start = None
        val_autoencoder_progress = 0
        val_autoencoder_progress_needed = None

        # Raw/Reduced criterion weights
        e = max(epoch - encoder_only_epochs - 1, 0) if enable_encoder_specific_training else epoch - 1

        criterion_weights_raw = e * raw_criterion_weight_inc + raw_criterion_weight
        criterion_weights_reduced = e * reduced_criterion_weight_inc + reduced_criterion_weight
        criterion_weights_norm = criterion_weights_raw + criterion_weights_reduced
        criterion_weights_raw /= criterion_weights_norm
        criterion_weights_reduced /= criterion_weights_norm

        for is_train in (True, False):
            gradients_thrown_out = None
            if is_train:
                print(f"\rEpoch {epoch}: Training...", end="")
                model.train()

                gradients_thrown_out = []
            else:
                print(f"\rEpoch {epoch}: Validating...", end="", flush=True)
                model.eval()

            for is_encoder_only in (True, False):
                if is_encoder_only and not enable_encoder_specific_training:
                    continue

                if is_train:
                    optimizer.zero_grad()

                    if is_encoder_only:
                        train_encoder_start = time.time()
                    else:
                        train_autoencoder_start = time.time()
                else:
                    if is_encoder_only:
                        val_encoder_start = time.time()
                    else:
                        val_autoencoder_start = time.time()

                if is_encoder_only:
                    loader = train_encoder_loader if is_train else val_encoder_loader

                    cos_sim_mode = True
                    nonlinear_monster = True

                    for pairs_batch, sim_batch in loader:
                        if cos_sim_mode:
                            predicted_sims = torch.cosine_similarity(model.encoder(pairs_batch[:, 0]),
                                                                     model.encoder(pairs_batch[:, 1]),
                                                                     dim=-1)

                            loss = encoder_criterion(predicted_sims, sim_batch).mean()
                        else:
                            zero = torch.tensor(0.)
                            sim_batch_plus = torch.max(sim_batch, zero)
                            sim_batch_minus = torch.max(-sim_batch, zero)
                            sim_batch_abs = torch.abs(sim_batch)
                            sim_batch_abs_sum = sim_batch_abs.sum()

                            L = encoder_criterion(model.encoder(pairs_batch[:, 0]),
                                                  model.encoder(pairs_batch[:, 1])).sum(dim=-1)

                            if nonlinear_monster:
                                sim_batch_is_pos = torch.heaviside(sim_batch, zero)
                                sim_batch_is_neg = torch.heaviside(-sim_batch, zero)

                                raw_loss = max_dissimilar_loss * ((sim_batch_is_pos * (L / max_dissimilar_loss) ** (
                                    (sim_batch_plus + peg))) + (sim_batch_is_neg * (
                                        (max_dissimilar_loss - L) / max_dissimilar_loss) ** (
                                                                    (sim_batch_minus + peg))))

                                loss = torch.where(torch.isclose(sim_batch_abs, zero), 0. * L + max_dissimilar_loss,
                                                   raw_loss)
                            else:
                                loss = sim_batch_plus * L + (peg + sim_batch_minus) * (
                                        max_dissimilar_loss - L) + center * (1. - sim_batch_abs - peg)

                            if not torch.isclose(sim_batch_abs_sum, zero):
                                loss = (sim_batch_abs * loss).sum() / sim_batch_abs_sum
                            else:
                                loss = loss.mean()

                        loss.backward()

                        if clipping_is_norm:
                            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                                 max_norm=gradient_norm_threshold,
                                                                 norm_type=clipping_norm_type,
                                                                 error_if_nonfinite=True)
                        else:
                            grads = torch.cat([torch.flatten(p.grad) for p in model.parameters()])
                            gradients_thrown_out.append(
                                np.mean(np.max([(grads - gradient_value_threshold) / grads, np.zeros(len(grads))],
                                               axis=0,
                                               where=grads > 0., initial=0.)))

                            nn.utils.clip_grad_value_(model.parameters(), gradient_value_threshold)

                        optimizer.step()

                        if is_train:
                            train_encoder_loss += loss.item()

                            if train_encoder_progress_needed is None:
                                train_encoder_progress_needed = len(train_encoder_loader)

                            train_encoder_progress += 1

                            datapoints_per_second = train_encoder_progress / (
                                    time.time() - train_encoder_start) * batch_size
                        else:
                            val_encoder_loss += loss.item()

                            if val_encoder_progress_needed is None:
                                val_encoder_progress_needed = len(val_encoder_loader)

                            val_encoder_progress += 1

                            datapoints_per_second = val_encoder_progress / (
                                    time.time() - val_encoder_start) * batch_size

                        dps_text = f"{datapoints_per_second:.2f} datapoints per second" if datapoints_per_second > 1. else f"{misc.timeformat(1. / datapoints_per_second)} per datapoint"

                        if is_train:
                            print(
                                f"\rEpoch {epoch}: Training (Encoder)... {train_encoder_progress / train_encoder_progress_needed:.1%} ETA {misc.timeformat(misc.timeleft(train_encoder_start, time.time(), train_encoder_progress, train_encoder_progress_needed))}, {dps_text} (Average criterion: {train_encoder_loss / train_encoder_progress:.5e})",
                                end="", flush=True)
                        else:
                            print(
                                f"\rEpoch {epoch}: Validating (Encoder)... {val_encoder_progress / val_encoder_progress_needed:.1%} ETA {misc.timeformat(misc.timeleft(val_encoder_start, time.time(), val_encoder_progress, val_encoder_progress_needed))}, {dps_text}",
                                end="", flush=True)

                elif not skip_decoder:
                    loader = train_autoencoder_loader if is_train else val_autoencoder_loader

                    for true_sequences in loader:
                        losses = None

                        if is_train:
                            losses = []

                        true_sequences = true_sequences.to(device)
                        true_sequences_sparse = torch.argmax(true_sequences, dim=-1)

                        true_sequences_reduced_sparse = torch.argmax(reduce_sequence(true_sequences), dim=-1)

                        outputs = model(true_sequences).to(device)
                        outputs_sparse = torch.argmax(outputs, dim=-1)
                        outputs_reduced = reduce_sequence(outputs, input_is_logs=True)

                        loss_raw_criterion = autoencoder_criterion_raw(outputs.transpose(1, 2), true_sequences_sparse)
                        loss_reduced_criterion = autoencoder_criterion_reduced(
                            outputs_reduced.transpose(1, 2),
                            true_sequences_reduced_sparse)

                        if loss_reduced_criterion < 0.0005:
                            loss_criterion = loss_raw_criterion
                        else:
                            loss_criterion = criterion_weights_raw * loss_raw_criterion + (
                                    criterion_weights_reduced * loss_reduced_criterion)

                        loss = loss_criterion

                        if is_train:
                            if torch.isfinite(loss):
                                losses.append(loss)
                            else:
                                print(f"WARNING: a non-finite loss was encountered ({loss}). It will not be considered")

                            train_autoencoder_loss += loss.item()
                            train_autoencoder_raw_criterion += loss_raw_criterion.item()
                            train_autoencoder_reduced_criterion += loss_reduced_criterion.item()
                            train_autoencoder_total_criterion += loss_criterion.item()

                            if loss_criterion.item() > train_autoencoder_largest_criterion:
                                train_autoencoder_largest_criterion = loss_criterion.item()
                        else:
                            val_autoencoder_loss += loss.item()
                            val_autoencoder_raw_criterion += loss_raw_criterion.item()
                            val_autoencoder_reduced_criterion += loss_reduced_criterion.item()
                            val_autoencoder_total_criterion += loss_criterion.item()

                        # Accuracy and Perfects
                        eq = torch.eq(true_sequences_sparse, outputs_sparse).to(dtype=torch.float32)
                        acc = eq.mean(dim=-1).mean(dim=-1).item()
                        prfct = eq.min(dim=-1).values.mean(dim=-1).item()

                        if is_train:
                            train_autoencoder_accuracy += acc
                            train_autoencoder_perfect_matches += prfct
                        else:
                            val_autoencoder_accuracy += acc
                            val_autoencoder_perfect_matches += prfct

                        B = true_sequences.shape[0]
                        for b in range(B):
                            if is_train:
                                if train_autoencoder_progress_needed is None:
                                    train_autoencoder_progress_needed = len(train_autoencoder_loader) * B
                            else:
                                if val_autoencoder_progress_needed is None:
                                    val_autoencoder_progress_needed = len(val_autoencoder_loader) * B

                            if not is_train and (val_autoencoder_example == "" or make_examples_csv):
                                list_true = string_from_onehots(true_sequences[b], list_mode=True)
                                list_output = string_from_onehots(outputs[b], list_mode=True)
                                str_true = "".join(list_true)
                                str_output = "".join(list_output)

                                if val_autoencoder_example == "":
                                    val_autoencoder_example = str_true
                                    val_autoencoder_autoencoded = str_output

                                if make_examples_csv:
                                    new_example = dict()
                                    new_example["Epoch"] = epoch
                                    new_example["Example"] = str_true
                                    new_example["AutoencodedExample"] = str_output
                                    new_example["Accuracy"] = np.mean([t == o for t, o in zip(list_true, list_output)])

                                    val_autoencoder_examples.append(new_example)

                            if not flatten_trees:
                                output_tokens = string_from_sparse(outputs_sparse[b], list_mode=True)

                                if pf.is_representation_valid("".join(output_tokens[0:-1]),
                                                              features=INDIVIDUALS_FEATURES):
                                    if is_train:
                                        train_autoencoder_valid += 1. / B
                                    else:
                                        val_autoencoder_valid += 1. / B

                            if is_train:
                                train_autoencoder_progress += 1

                                datapoints_per_second = train_autoencoder_progress / (
                                        time.time() - train_autoencoder_start)
                            else:
                                val_autoencoder_progress += 1

                                datapoints_per_second = val_autoencoder_progress / (time.time() - val_autoencoder_start)

                            dps_text = f"{datapoints_per_second:.2f} datapoints per second" if datapoints_per_second > 1. else f"{misc.timeformat(1. / datapoints_per_second)} per datapoint"

                            if is_train:
                                gradient_norms_text = f"{np.mean(gradients_thrown_out):.2%}" if len(
                                    gradients_thrown_out) > 0 else "N/A"
                                print(
                                    f"\rEpoch {epoch}: Training (Autoencoder)... {train_autoencoder_progress / train_autoencoder_progress_needed:.1%} ETA {misc.timeformat(misc.timeleft(train_autoencoder_start, time.time(), train_autoencoder_progress, train_autoencoder_progress_needed))}, {dps_text} (Average criterion: {train_autoencoder_total_criterion / train_autoencoder_progress * batch_size:.3f} ({criterion_weights_raw:.2f}*{train_autoencoder_raw_criterion / train_autoencoder_progress * batch_size:.3f}+{criterion_weights_reduced:.2f}*{train_autoencoder_reduced_criterion / train_autoencoder_progress * batch_size:.3f}), Largest criterion: {train_autoencoder_largest_criterion:.3f}, Gradients' norms lost due to clipping: {gradient_norms_text})",
                                    end="", flush=True)
                            else:
                                print(
                                    f"\rEpoch {epoch}: Validating (Autoencoder)... {val_autoencoder_progress / val_autoencoder_progress_needed:.1%} ETA {misc.timeformat(misc.timeleft(val_autoencoder_start, time.time(), val_autoencoder_progress, val_autoencoder_progress_needed))}, {dps_text}",
                                    end="", flush=True)

                        if is_train:
                            losses = torch.stack(losses)
                            losses.mean().backward()

                            # zero out NaN gradients
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad[torch.isnan(p.grad)] = 0.

                            if clipping_is_norm:
                                grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                                     max_norm=gradient_norm_threshold,
                                                                     norm_type=clipping_norm_type,
                                                                     error_if_nonfinite=True)

                                if grad_norm > 0.:
                                    norm = grad_norm.item()
                                    gradients_thrown_out.append(max((norm - gradient_norm_threshold) / norm, 0))
                            else:
                                grads = torch.cat([torch.flatten(p.grad) for p in model.parameters()])
                                gradients_thrown_out.append(
                                    np.mean(np.max([(grads - gradient_value_threshold) / grads, np.zeros(len(grads))],
                                                   axis=0,
                                                   where=grads > 0., initial=0.)))

                                nn.utils.clip_grad_value_(model.parameters(), gradient_value_threshold)

                            optimizer.step()

        if enable_encoder_specific_training:
            l_e_t = train_encoder_progress
            l_e_v = val_encoder_progress

            train_encoder_loss /= l_e_t
            val_encoder_loss /= l_e_v

        l_d_t = train_autoencoder_progress
        L_d_t = train_autoencoder_progress / batch_size
        l_d_v = val_autoencoder_progress
        L_d_v = val_autoencoder_progress / batch_size

        if not skip_decoder:
            train_autoencoder_loss /= L_d_t
            train_autoencoder_raw_criterion /= L_d_t
            train_autoencoder_reduced_criterion /= L_d_t
            train_autoencoder_total_criterion /= L_d_t

            train_autoencoder_accuracy /= L_d_t
            train_autoencoder_valid /= L_d_t
            train_autoencoder_perfect_matches /= L_d_t

        de_facto_raw_weight = (train_autoencoder_total_criterion - train_autoencoder_reduced_criterion) / (
                train_autoencoder_raw_criterion - train_autoencoder_reduced_criterion) if train_autoencoder_raw_criterion > train_autoencoder_reduced_criterion else 1.
        de_facto_reduced_weight = 1. - de_facto_raw_weight

        if not skip_decoder:
            val_autoencoder_loss /= L_d_v
            val_autoencoder_raw_criterion /= L_d_v
            val_autoencoder_reduced_criterion /= L_d_v
            val_autoencoder_total_criterion = de_facto_raw_weight * val_autoencoder_raw_criterion + de_facto_reduced_weight * val_autoencoder_reduced_criterion

            val_autoencoder_accuracy /= L_d_v
            val_autoencoder_valid /= L_d_v
            val_autoencoder_perfect_matches /= L_d_v

        current_encoder_lr = optimizer.param_groups[0]['lr']
        current_autoencoder_lr = optimizer.param_groups[1]['lr']

        # encoder_scheduler.step(val_encoder_loss)

        if val_autoencoder_accuracy > 0.:
            autoencoder_scheduler.step(val_autoencoder_accuracy)

        if skip_decoder:
            train_autoencoder_start = val_encoder_start
            val_autoencoder_start = time.time()
        elif not enable_encoder_specific_training:
            train_encoder_start = train_autoencoder_start
            val_encoder_start = train_autoencoder_start

        print(
            f"\rEpoch {epoch}: Encode/Autoencode/Valid/Accuracy/Perfects: (Train: {train_encoder_loss:.5e}/{train_autoencoder_total_criterion:.4f} ({de_facto_raw_weight:.2f}*{train_autoencoder_raw_criterion:.3f}+{de_facto_reduced_weight:.2f}*{train_autoencoder_reduced_criterion:.3f})/{train_autoencoder_valid:.2%}/{train_autoencoder_accuracy:.2%}/{train_autoencoder_perfect_matches:.2%}) (Val: {val_encoder_loss:.5e}/{val_autoencoder_total_criterion:.4f} ({de_facto_raw_weight:.2f}*{val_autoencoder_raw_criterion:.3f}+{de_facto_reduced_weight:.2f}*{val_autoencoder_reduced_criterion:.3f})/{val_autoencoder_valid:.2%}/{val_autoencoder_accuracy:.2%}/{val_autoencoder_perfect_matches:.2%}) (Total data: {train_autoencoder_set.total_datapoints}, {val_autoencoder_set.total_datapoints}) LR (En/De): {current_encoder_lr:.0e}/{current_autoencoder_lr:.0e} Took {misc.timeformat(time.time() - train_encoder_start)} ({misc.timeformat(train_autoencoder_start - train_encoder_start)}, {misc.timeformat(val_encoder_start - train_autoencoder_start)}, {misc.timeformat(val_autoencoder_start - val_encoder_start)}, {misc.timeformat(time.time() - val_autoencoder_start)})"
        )

        new_row = dict()
        new_row["Epoch"] = epoch
        new_row["Criterion_Weight_Raw"] = de_facto_raw_weight
        new_row["Criterion_Weight_Reduced"] = de_facto_reduced_weight
        new_row["Train_Autoencoder_TotalDatapoints"] = train_autoencoder_set.total_datapoints
        new_row["Train_Encoder_Loss"] = train_encoder_loss

        if not skip_decoder:
            new_row["Train_Autoencoder_Loss"] = train_autoencoder_loss
            new_row["Train_Autoencoder_Total_Criterion"] = train_autoencoder_total_criterion
            new_row["Train_Autoencoder_Raw_Criterion"] = train_autoencoder_raw_criterion
            new_row["Train_Autoencoder_Reduced_Criterion"] = train_autoencoder_reduced_criterion
            new_row["Train_Autoencoder_Valid"] = train_autoencoder_valid
            new_row["Train_Autoencoder_Accuracy"] = train_autoencoder_accuracy
            new_row["Train_Autoencoder_Perfects"] = train_autoencoder_perfect_matches

        new_row["Train_Encoder_LR"] = current_encoder_lr
        new_row["Train_Autoencoder_LR"] = current_autoencoder_lr

        new_row["Val_Autoencoder_TotalDatapoints"] = val_autoencoder_set.total_datapoints
        new_row["Val_Encoder_Loss"] = val_encoder_loss

        if not skip_decoder:
            new_row["Val_Autoencoder_Loss"] = val_autoencoder_loss
            new_row["Val_Autoencoder_Total_Criterion"] = val_autoencoder_total_criterion
            new_row["Val_Autoencoder_Raw_Criterion"] = val_autoencoder_raw_criterion
            new_row["Val_Autoencoder_Reduced_Criterion"] = val_autoencoder_reduced_criterion
            new_row["Val_Autoencoder_Valid"] = val_autoencoder_valid
            new_row["Val_Autoencoder_Accuracy"] = val_autoencoder_accuracy
            new_row["Val_Autoencoder_Perfects"] = val_autoencoder_perfect_matches

            new_row["Example"] = val_autoencoder_example
            new_row["AutoencodedExample"] = val_autoencoder_autoencoded

            if make_examples_csv:
                if len(df_examples) > 0:
                    df_examples = pd.concat(
                        [df_examples, pd.DataFrame(val_autoencoder_examples, index=range(len(val_autoencoder_examples)))],
                        ignore_index=True)
                else:
                    df_examples = pd.DataFrame(val_autoencoder_examples, index=range(len(val_autoencoder_examples)))

                df_examples.sort_values(by=["Example", "Epoch"], inplace=True)

                df_examples.to_csv(path_or_buf=f"{folder_name}/examples.csv", index=False)

        if len(df) > 0:
            df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        else:
            df = pd.DataFrame(new_row, index=[0])

        df.to_csv(path_or_buf=f"{folder_name}/log.csv", index=False)

        torch.save(model.state_dict(), f"{folder_name}/model_epoch{epoch}.pth")

        if val_autoencoder_perfect_matches >= 1.:
            if perfect_epochs_in_a_row >= val_autoencoder_perfects_100_patience:
                print(
                    f"The model has reached 100% perfect matches in the validation set for {val_autoencoder_perfects_100_patience} epoch(s) in a row, and has stopped training")
                break

            perfect_epochs_in_a_row += 1
        else:
            perfect_epochs_in_a_row = 0

        if epoch < num_epochs:
            print(f"Refreshing training and validation set for Epoch {epoch + 1}...", end="")
            train_autoencoder_set.refresh_data()

            if enable_encoder_specific_training:
                train_encoder_set.refresh_data()

            val_autoencoder_set.refresh_data()

            if enable_encoder_specific_training:
                val_encoder_set.refresh_data()


def reward_model_df_is_wide(df):
    return np.any([misc.begins_with(column, "Fitness_") for column in df.columns])


def reward_model_df_wide_to_long(df):
    result = pd.wide_to_long(df, stubnames="Fitness", sep="_", i="Individual", j="Seed").reset_index()
    result.dropna(inplace=True)
    result.sort_values(by=["Individual", "Seed"], inplace=True, ignore_index=True)

    return result


def generate_reward_model_file(batch_size=8,
                               additional_constants_per_batch=4,
                               num_batches=4000,
                               long=True,
                               seeds_per_batch=32,
                               number_of_possible_seeds=128,
                               explicit_seeds=None,
                               max_depth=8,
                               flatten_trees=True,
                               fill_trees=True,
                               features_weight_in_full_trees=5,
                               individuals_seed=None,
                               num_workers=5,
                               vocab=None,
                               warehouse_settings=None,
                               verbose=2):
    if flatten_trees and not fill_trees:
        raise ValueError("flatten_trees=True and fill_trees=False is not implemented")

    if warehouse_settings is None:
        warehouse_settings = dfjss.WarehouseSettings()

    autoencoder_dataset = AutoencoderDataset(size=batch_size,
                                             max_depth=max_depth,
                                             flatten_trees=flatten_trees,
                                             fill_trees=fill_trees,
                                             refresh_rate=1.,
                                             features_weight_in_full_trees=features_weight_in_full_trees,
                                             rng_seed=individuals_seed)

    autoencoder_constant_dataset = AutoencoderDataset(size=additional_constants_per_batch,
                                                      max_depth=max_depth,
                                                      flatten_trees=flatten_trees,
                                                      fill_trees=fill_trees,
                                                      refresh_rate=1.,
                                                      features_weight_in_full_trees=0.,
                                                      rng_seed=individuals_seed)

    fitness_log = None
    fitness_of_constant = None
    fitness_columns = None
    batch_n = 0
    while num_batches is None or (num_batches is not None and batch_n <= num_batches):
        try:
            evaluating_constant_individuals = batch_n <= 0

            if evaluating_constant_individuals:
                print("Evaluating fitness of constant individuals...")
            else:
                print(f"Batch {batch_n}{f' of {num_batches}' if num_batches is not None else ''}")

            start = time.time()

            individuals = []

            I = 1 if evaluating_constant_individuals else batch_size

            for i in range(I):
                if flatten_trees:
                    tree = pf.representation_to_priority_function_tree("({0.0}+{0.0})", features=INDIVIDUALS_FEATURES)
                    tree.fill(max_depth)

                    if not evaluating_constant_individuals:
                        tree.set_from_flattened(
                            string_from_onehots(autoencoder_dataset[i], vocab=vocab, list_mode=True))
                else:
                    if not evaluating_constant_individuals:
                        tree = pf.representation_to_priority_function_tree(autoencoder_dataset[i],
                                                                           features=INDIVIDUALS_FEATURES)
                    else:
                        tree = pf.representation_to_priority_function_tree("({0.0}+{0.0})",
                                                                           features=INDIVIDUALS_FEATURES)

                individuals.append(tree)

            gen_algo_settings = genetic.GeneticAlgorithmSettings()
            gen_algo_settings.features = INDIVIDUALS_FEATURES
            gen_algo_settings.warehouse_settings = warehouse_settings
            gen_algo_settings.multiprocessing_processes = num_workers
            gen_algo_settings.population_size = I
            gen_algo_settings.total_steps = 1
            gen_algo_settings.number_of_possible_seeds = number_of_possible_seeds
            gen_algo_settings.number_of_simulations_per_individual = number_of_possible_seeds if evaluating_constant_individuals else seeds_per_batch
            gen_algo_settings.simulations_seeds = explicit_seeds
            gen_algo_settings.save_logs_csv = False

            gen_algo = genetic.GeneticAlgorithm(settings=gen_algo_settings)
            gen_algo.population = individuals

            gen_algo_result = gen_algo.run_genetic_algorithm(sort_fitness_log=False, verbose=verbose)

            if fitness_columns is None:
                fitness_columns = [f"Fitness_{s}" for s in gen_algo_settings.simulations_seeds]

            fitness_log = gen_algo_result.fitness_log

            if evaluating_constant_individuals:
                fitness_of_constant = fitness_log.loc[0, fitness_columns].copy()
            else:
                # add random, constant individuals to the fitness log
                data = defaultdict(list)

                for j in range(additional_constants_per_batch):
                    constant_tree = pf.representation_to_priority_function_tree("({0.0}+{0.0})",
                                                                                features=INDIVIDUALS_FEATURES)

                    constant_tree.fill(max_depth)

                    constant_tree.set_from_flattened(
                        string_from_onehots(autoencoder_constant_dataset[j], vocab=vocab, list_mode=True))

                    data["Individual"].append(repr(constant_tree))

                    for c in fitness_columns:
                        data[c].append(fitness_of_constant[c])

                fitness_log = pd.concat([fitness_log, pd.DataFrame(data)])

            if flatten_trees:
                fitness_log["Individual"] = [ind.replace("(", "").replace(")", "") for ind in fitness_log["Individual"]]

            fitness_log.drop(columns=["Fitness", "Phenotype"], inplace=True)

            fitness_log["IsConstant"] = [int(all_features_are_constants(ind)) for ind in fitness_log["Individual"]]

            if not evaluating_constant_individuals and len(fitness_log) != (
                    batch_size + additional_constants_per_batch):
                raise Exception(
                    f"Length of fitness log {len(fitness_log)} does not match expected size {batch_size + additional_constants_per_batch}")

            if long:
                fitness_log = reward_model_df_wide_to_long(fitness_log)

            if os.path.exists(REWARDMODEL_FILENAME):
                old_fitness_log = pd.read_csv(REWARDMODEL_FILENAME)

                if long and reward_model_df_is_wide(old_fitness_log):
                    old_fitness_log = reward_model_df_wide_to_long(old_fitness_log)

                if not long:
                    columns_old = set(old_fitness_log.columns)
                    columns_current = set(fitness_log.columns)

                    if columns_old != columns_current:
                        print("Dataset on disk's columns and the computed dataset have different columns")

                        if columns_old.issubset(columns_current):
                            print(
                                f"Computed dataset has these additional columns: {columns_current.difference(columns_old)}")
                        else:
                            print(
                                f"Dataset on disk has these additional columns: {columns_old.difference(columns_current)}")

                        print("Datasets will be merged anyway")

                fitness_log = pd.concat([old_fitness_log, fitness_log])

            fitness_log.to_csv(path_or_buf=REWARDMODEL_FILENAME, index=False, mode='w')

            print(f"\rTook {misc.timeformat(time.time() - start)}", flush=True)

            print("Refreshing dataset...")
            autoencoder_dataset.refresh_data()
            if not evaluating_constant_individuals:
                autoencoder_constant_dataset.refresh_data()

            batch_n += 1

        except KeyboardInterrupt:
            print("\nData generation was manually interrupted")

            if fitness_log is None:
                print("Fitness log is None")
            break

    return fitness_log


def end_with_eos(string):
    return string if string.endswith("EOS") else string + "EOS"


class RewardModelDataset(data.Dataset):
    def __init__(self, autoencoder, autoencoder_folder, force_seeds=None, anti_decode=False,
                 refresh_strength=0.002, refresh_max_attempts=5, rng_seed=100,
                 verbose=1):
        """
        :type autoencoder: IndividualFeedForwardAutoEncoder

        :param autoencoder:
        :param autoencoder_folder:
        :param force_seeds:
        :param verbose:
        """
        super().__init__()

        self.autoencoder = autoencoder

        try:
            self.df = pd.read_csv(REWARDMODEL_FILENAME)
        except OSError as os_error:
            raise OSError(f"Could not read {REWARDMODEL_FILENAME}. Error given: {os_error}")

        df_is_wide = reward_model_df_is_wide(self.df)

        if df_is_wide:
            self.df = reward_model_df_wide_to_long(self.df)

        # mean fitness, integrating out seed
        df_seedmean = self.df.groupby(["Individual"]).mean().reset_index()
        df_seedmean["Seed"] = -1

        self.df = pd.concat([self.df, df_seedmean], ignore_index=True)

        if force_seeds is not None:
            self.seeds = torch.tensor(np.unique(force_seeds))
            self.df = self.df[[(seed in self.seeds) for seed in self.df.loc[:, "Seed"]]].reset_index()
        else:
            self.seeds = torch.tensor(np.unique(self.df["Seed"])) # self.df.loc[self.df["Seed"] != -1, "Seed"]

        self.N = len(self.df)

        # make sure df is sorted
        self.df.sort_values(by=["Individual", "Seed"], inplace=True)

        self.raw_individuals = np.unique(self.df["Individual"])

        precompute_table_path = f"{autoencoder_folder}/{AUTOENCODER_PRECOMPUTE_TABLE_FILENAME}"
        self.precompute_table = autoencoder.get_precompute_table(self.raw_individuals,
                                                                 precompute_table_path,
                                                                 anti_decode=anti_decode,
                                                                 verbose=verbose)

        self.refresh_strength = refresh_strength
        self.refresh_max_attempts = refresh_max_attempts

        self.rng = torch.Generator()
        self.rng.manual_seed(rng_seed)

        self._original_individuals_data = torch.stack([self.precompute_table[end_with_eos(ind)] for ind in self.raw_individuals])
        self._original_individuals_sparse = torch.stack([sparse_sequence(end_with_eos(ind)) for ind in self.raw_individuals])

        self._individuals_data = self._original_individuals_data.clone()

        self.individual_sequence_length = self._individuals_data.shape[1]

        self.individual_to_data = dict([(raw, tens) for raw, tens in zip(self.raw_individuals, self._individuals_data)])

        self.num_seeds = len(self.seeds)

        self.seeds_data = torch.tensor(self.df.loc[:, "Seed"].to_numpy(dtype="int"))

        self.rewards_data = torch.tensor(self.df.loc[:, "Fitness"].to_numpy(dtype="float32"))

    def summary(self):
        S = self.N
        counter = Counter([int(seed.item()) for seed in self.seeds_data])
        perpl = np.exp(np.sum([-freq / S * np.log(freq / S) for seed, freq in counter.most_common()]))

        print("Grouping rewards by seed to calculate baseline L1 and L2... ", end="")

        rewards_dict = defaultdict(list)

        for i in range(self.N):
            rewards_dict[self.seeds_data[i].item()].append(self.rewards_data[i].item())

        print("Done")

        for key in rewards_dict.keys():
            rewards_dict[key] = np.array(rewards_dict[key])

        base_l2 = dict([(key, np.round(np.var(rewards_dict[key]), decimals=2)) for key in rewards_dict.keys()])
        base_l1 = dict([(key, np.round(np.abs(rewards_dict[key] - np.mean(rewards_dict[key])).mean(), decimals=2)) for key in rewards_dict.keys()])

        longdash = '------------------------------------'
        result = [longdash, "Reward Model Dataset",
                  f"Number of Individuals: {len(self.raw_individuals)}",
                  f"Number of Seeds: {self.num_seeds}",
                  f"Sample size (number of individual-seed pairs): {self.N}",
                  f"Counter of seeds: {counter}",
                  f"Perplexity of seed distribution: {perpl:.3f}",
                  f"Unconditional L2 loss: {base_l2}",
                  f"Unconditional L1 loss: {base_l1}",
                  longdash]

        return "\n".join(result)

    def refresh_data(self):
        if True:
            self._individuals_data = perturb(self._original_individuals_data, strength=self.refresh_strength, generator=self.rng)
        else:
            I = self._original_individuals_data.shape[0]
            start = time.time()
            for i in range(I):
                attempt = 1
                candidate = None
                while attempt <= self.refresh_max_attempts:
                    candidate = perturb(self._original_individuals_data[i], strength=self.refresh_strength, generator=self.rng)

                    candidate_sparse = self.autoencoder.decoder(candidate).argmax(dim=-1)

                    if torch.equal(candidate_sparse, self._original_individuals_sparse[i]):
                        break
                    else:
                        attempt += 1

                self._individuals_data[i] = candidate
                print(f"\rRefreshing data... {misc.timeformat(misc.timeleft(start, time.time(), i + 1, I))}", end="",
                      flush=True)

    def get_individual(self, idx):
        return self.individual_to_data[self.df.loc[idx, "Individual"]]

    def __iter__(self):
        return self.data_iterator()

    def __getitem__(self, idx):
        return self.get_individual(idx), self.seeds_data[idx], self.rewards_data[idx]

    def __len__(self):
        return self.N

    def data_iterator(self):
        for i in range(len(self)):
            yield self[i]


def perturb(encoding, strength=0.01, p=0., generator=None):
    shape = tuple(encoding.shape)
    with torch.no_grad():
        noise = torch.empty(size=shape).uniform_(-strength, strength, generator=generator)

        result = torch.clamp(encoding + noise, min=-1, max=1)

        if p > 0.:
            peg = torch.bernoulli(torch.full_like(encoding, fill_value=p), generator=generator)

            result = (1. - peg) * result + peg * encoding

        return result


class RewardModel(nn.Module):
    autoencoder: IndividualFeedForwardAutoEncoder

    def __init__(self, input_size,
                 seeds,
                 embedding_dim=128,
                 num_layers=2,
                 layer_widths=(1024,),
                 layer_dropout=0.1,
                 layers_are_residual=False,
                 reward_activation="elu"):
        super().__init__()

        if not isinstance(layer_widths, tuple):
            layer_widths = (layer_widths,)

        num_widths = len(layer_widths)

        self.register_buffer("input_size", torch.tensor(input_size))

        self.register_buffer("seeds", seeds.clone().detach())

        self.seed_to_index = None
        self.make_seed_to_index()

        self.register_buffer("num_seeds", torch.tensor([len(self.seeds)]))

        if embedding_dim > 0:
            self.seed_embedding = nn.Embedding(embedding_dim=embedding_dim,
                                               num_embeddings=self.num_seeds + 1,
                                               padding_idx=0,
                                               device=device)
        else:
            self.seed_embedding = None

        self.reward_activation = reward_activation

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.layer_activations = nn.ModuleList()
        self.layer_dropout = nn.Dropout(layer_dropout)

        self.register_buffer("layers_are_residual", torch.tensor([layers_are_residual]))

        for i in range(num_layers):
            if i == 0:
                in_width = input_size + max(embedding_dim, 0)
                out_width = layer_widths[0]
            else:
                in_width = layer_widths[(i - 1) % num_widths]
                out_width = layer_widths[i % num_widths]

            self.layers.append(nn.Linear(in_features=in_width, out_features=out_width, device=device))
            self.layer_norms.append(nn.BatchNorm1d(out_width, device=device))
            self.layer_activations.append(nn.PReLU(device=device))

        self.last_layer_to_reward = nn.Linear(in_features=self.layers[-1].out_features, out_features=1, device=device)

    def make_seed_to_index(self, force=False):
        if force:
            self.seed_to_index = None

        if self.seed_to_index is not None:
            raise Exception("seed_to_index already created")

        S = list(self.seeds)
        if -1 in S:
            S.pop(S.index(-1))

        self.seed_to_index = dict([(seed.item(), torch.tensor(i + 1)) for i, seed in enumerate(S)])
        self.seed_to_index[-1] = torch.tensor(0)

    def get_layers_are_residual(self):
        return self.layers_are_residual.item()

    def import_state(self, path):
        self.load_state_dict(torch.load(path))

        self.make_seed_to_index(True)

        self.eval()
        print("Note: reward model has been set to eval mode")

    def forward(self, x, seeds):
        is_batch = x.dim() == 2

        if self.seed_embedding is not None:
            if is_batch:
                embed = self.seed_embedding(
                    torch.stack([self.seed_to_index.get(seed.item(), torch.tensor(0)) for seed in seeds], dim=0))
            else:
                embed = self.seed_embedding(self.seed_to_index.get(seeds.item(), torch.tensor(0)))

            y = torch.cat([x, embed], dim=-1)
        else:
            y = x

        for i, layer, norm, activ in zip(range(len(self.layers)), self.layers, self.layer_norms,
                                         self.layer_activations):
            y_transformed = activ(
                (norm(layer(y)) if is_batch else norm(layer(y).unsqueeze(0)).squeeze(0)) +
                (y if (self.get_layers_are_residual() and layer.in_features == layer.out_features) else 0.))

            y = (self.layer_dropout(y_transformed) if i > 0 else y_transformed)

        y = self.last_layer_to_reward(y)

        if self.reward_activation == "exp":
            y = torch.exp(y)
        if self.reward_activation == "elu":
            y = torch.nn.functional.elu(y) + 1.

        return y.squeeze(-1) if is_batch else y

    def count_parameters(self, learnable=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad == learnable)

    def summary(self):
        longdash = '------------------------------------'
        result = [longdash, "Reward Model",
                  f"Input size: {self.input_size}",
                  f"Number of Seeds: {self.num_seeds.item()}",
                  f"Seed Embedding size: {self.seed_embedding.embedding_dim if self.seed_embedding is not None else 'N/A'}",
                  f"Hidden Layers: {len(self.layers)} ({', '.join([str(layer.in_features) + '->' + str(layer.out_features) for layer in self.layers])})",
                  f"Layers of same size are residual: {'yes' if self.get_layers_are_residual() else 'no'}",
                  f"Layer Dropout: {self.layer_dropout.p}",
                  f"Final Activation: {self.reward_activation}",
                  f"Number of parameters: (Learnable: {self.count_parameters()}, Fixed: {self.count_parameters(False)})",
                  longdash]

        return "\n".join(result)


def train_reward_model(model,
                       dataset,
                       num_epochs=50,
                       batch_size=64,
                       val_split=0.2,
                       weight_decay=0.001,
                       loss_weights=(0., 1.0, 0.),
                       loss_weights_inc=(0., 0., 0.),
                       loss_names=("L1", "SmoothL1", "L2"),
                       gradient_clipping=1.,
                       smooth_l1_beta=50.,
                       individual_perturbation=1500,
                       contributions_degree=2):
    """
    :type model: RewardModel
    :type dataset: RewardModelDataset

    :param model:
    :param dataset:
    :param num_epochs:
    :param batch_size:
    :param val_split:
    :param weight_decay:
    :param loss_weights:
    :param loss_weights_inc:
    :param loss_names:
    :param gradient_clipping:
    :param smooth_l1_beta:
    :param contributions_degree:
    :return:
    """
    num_losses = len(loss_weights)
    assert len(loss_weights) == len(loss_weights_inc)
    assert len(loss_weights) == len(loss_names)
    assert len(loss_weights) == 3

    folder_name = datetime.datetime.now().strftime(f'REWARD MODEL %Y-%m-%d %H-%M-%S')

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print(f"Model(s) will be saved in \"{folder_name}\"")

    with open(f"{folder_name}/model_summary.txt", "w") as model_summary_file:
        model_summary_file.write(model.summary())

    with open(f"{folder_name}/dataset_summary.txt", "w") as dataset_summary_file:
        dataset_summary_file.write(dataset.summary())

    df_cols = ["Epoch", "Loss_Weight_PerReward", "Loss_Weight_MeanReward"]

    for tv in ("Train", "Val"):
        for q in ("Loss", "PerReward_Loss", "MeanReward_Loss"):
            df_cols.append(f"{tv}_{q}")

        if tv == "Train":
            df_cols.append("Train_LR")

    df = pd.DataFrame(columns=df_cols)

    # Train-Val split

    train_set, val_set = data.random_split(dataset=dataset, lengths=[1. - val_split, val_split])

    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True
    )

    # First-layer Contributions

    weights_for_contributions_dict = dict()

    weights_for_contributions_dict["Individual"] = model.layers[0].weight[:, 0:model.input_size]
    weights_for_contributions_dict["Seed_Embedding"] = model.layers[0].weight[:,
                                                       model.input_size:model.layers[0].weight.shape[1]]

    print(
        f"(Sanity check) Size of contribution weights matrices: {[(key, value.shape) for key, value in weights_for_contributions_dict.items()]}")

    def get_contributions():
        result = dict()
        with torch.no_grad():
            norm = torch.pow(torch.abs(model.layers[0].weight), contributions_degree).sum()

            for key, value in weights_for_contributions_dict.items():
                result[key] = (torch.pow(torch.abs(value), contributions_degree).sum() / norm).item()

        return result

    def contributions_format(c):
        return "[" + ", ".join([f"{key}: {value:.2%}" for key, value in c.items()]) + "]"

    print(
        f"(Sanity check) Initial L{contributions_degree} contributions: {contributions_format(get_contributions())}")

    # Weight decay

    weights = []
    biases = []

    for name, param in model.named_parameters():
        if "weight" in name:
            weights.append(param)
        else:
            biases.append(param)

    criterion_funcs = [nn.L1Loss(), nn.SmoothL1Loss(beta=smooth_l1_beta), nn.MSELoss()]

    optimizer = optim.Adam([{'params': weights, 'weight_decay': weight_decay},
                            {'params': biases, 'weight_decay': 0.}], lr=0.001)

    print(f"Weight decay = {weight_decay}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min",
                                                     patience=5, cooldown=0, factor=0.1 ** 0.25,
                                                     verbose=False)

    for epoch in range(1, num_epochs + 1):
        # Training
        train_loss = 0.
        train_criterions = np.zeros(shape=num_losses)

        train_start = None
        train_progress = 0
        train_progress_needed = None

        # Validation
        val_loss = 0.
        val_criterions = np.zeros(shape=num_losses)

        val_start = None
        val_progress = 0
        val_progress_needed = None

        # Criterion weights
        criterion_weights = torch.tensor(data=loss_weights, requires_grad=False, device=device)
        criterion_weights += (epoch - 1) * torch.tensor(data=loss_weights_inc, requires_grad=False, device=device)
        criterion_weights /= torch.sum(criterion_weights)

        def running_loss_and_its_components(train, norm):
            if train:
                total_loss = train_loss / train_progress if norm else train_loss
                criterions = train_criterions / train_progress if norm else train_criterions
            else:
                total_loss = val_loss / val_progress if norm else val_loss
                criterions = val_criterions / val_progress if norm else val_criterions

            return f"{total_loss:.3f} (" + "+".join(
                [f"{criterion_weights[i]:.2f}*{criterions[i]:.3f}" for i in range(num_losses)]) + ")"

        for is_train in (True, False):
            if is_train:
                print(f"\rEpoch {epoch}: Training...", end="")
                model.train()

                train_start = time.time()

                if model.seed_embedding is not None:
                    with torch.no_grad():
                        embed_params_before = model.seed_embedding.weight.data.clone()
            else:
                print(f"\rEpoch {epoch}: Validating...", end="", flush=True)
                model.eval()

                val_start = time.time()

            loader = train_loader if is_train else val_loader

            for individual_batch, seed_batch, reward_batch in loader:
                B = individual_batch.shape[0]

                if is_train:
                    if train_progress_needed is None:
                        train_progress_needed = len(train_loader)
                else:
                    if val_progress_needed is None:
                        val_progress_needed = len(val_loader)

                if is_train:
                    optimizer.zero_grad()

                #if individual_perturbation is not None:
                    #individual_batch = perturb(individual_batch, v=individual_perturbation)

                predicted_rewards = model(individual_batch, seed_batch)

                l = torch.stack([crit(predicted_rewards, reward_batch) for crit in criterion_funcs], dim=-1)

                loss = torch.linalg.vecdot(l, criterion_weights)

                if is_train:
                    train_progress += 1

                    batches_per_second = train_progress / (time.time() - train_start)

                    train_loss += loss.item()
                    train_criterions += l.numpy(force=True)
                else:
                    val_progress += 1

                    batches_per_second = val_progress / (time.time() - val_start)

                    val_loss += loss.item()
                    val_criterions += l.numpy(force=True)

                dps_text = f"{batches_per_second:.2f} batches per second ({batches_per_second * B:.2f} datapoints/s)" if batches_per_second > 1. else f"{misc.timeformat(1. / batches_per_second)} per batch ({misc.timeformat(1. / (batches_per_second * B))} per datapoint)"

                if is_train:
                    print(
                        f"\rEpoch {epoch}: Training... {train_progress} / {train_progress_needed} ({train_progress / train_progress_needed:.1%}) ETA {misc.timeformat(misc.timeleft(train_start, time.time(), train_progress, train_progress_needed))}, {dps_text}, Average criterion: {running_loss_and_its_components(True, True)}",
                        end="", flush=True)
                else:
                    print(
                        f"\rEpoch {epoch}: Validating... {val_progress} / {val_progress_needed} ({val_progress / val_progress_needed:.1%}) ETA {misc.timeformat(misc.timeleft(val_start, time.time(), val_progress, val_progress_needed))}, {dps_text}, Average criterion: {running_loss_and_its_components(False, True)}",
                        end="", flush=True)

                if is_train:
                    loss.mean().backward()

                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                         max_norm=gradient_clipping,
                                                         error_if_nonfinite=True)

                    optimizer.step()

        train_norm = train_progress
        val_norm = val_progress

        train_loss /= train_norm
        train_criterions /= train_norm

        val_loss /= val_norm
        val_criterions /= val_norm

        current_lr = optimizer.param_groups[0]['lr']

        if model.seed_embedding is not None:
            with torch.no_grad():
                embed_param_shift = "{:.4f}".format(
                    torch.abs(model.seed_embedding.weight.data - embed_params_before).sum().item())
        else:
            embed_param_shift = "N/A"

        contribs = get_contributions()

        print(
            f"\rEpoch {epoch}: Loss({'+'.join(loss_names)}): (Train: {running_loss_and_its_components(True, False)}) (Val: {running_loss_and_its_components(False, False)}) LR: {current_lr:.0e}, L{contributions_degree} Contributions: {contributions_format(contribs)}, Embedding's parameter shift: {embed_param_shift}, Took {misc.timeformat(time.time() - train_start)} ({misc.timeformat(val_start - train_start)}, {misc.timeformat(time.time() - val_start)})"
        )

        scheduler.step(val_loss)

        new_row = dict()
        new_row["Epoch"] = epoch
        for i in range(num_losses):
            new_row[f"Loss_Weight_{loss_names[i]}"] = criterion_weights[i].item()

        new_row["Train_Loss"] = train_loss
        for i in range(num_losses):
            new_row[f"Train_{loss_names[i]}_Loss"] = train_criterions[i]

        new_row["Train_LR"] = current_lr

        new_row["Val_Loss"] = val_loss
        for i in range(num_losses):
            new_row[f"Val_{loss_names[i]}_Loss"] = val_criterions[i]

        for key, value in contribs.items():
            new_row[f"L{contributions_degree}_Contribution_{key}"] = value

        if len(df) > 0:
            df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        else:
            df = pd.DataFrame(new_row, index=[0])

        df.to_csv(path_or_buf=f"{folder_name}/log.csv", index=False)

        torch.save(model.state_dict(), f"{folder_name}/model_epoch{epoch}.pth")

        start = time.time()
        dataset.refresh_data()
        #print(f"Took {misc.timeformat(time.time() - start)}")


def optimize_reward(model,
                    autoencoder,
                    dataset,
                    seed_to_optimize=-1,
                    mode="adam",
                    init="best_of_dataset",
                    max_iters=2000,
                    iters_per_print=25,
                    gradient_max_norm=1.,
                    rng_seed=123,
                    verbose=1):
    """
    :type model: RewardModel
    :type autoencoder: IndividualFeedForwardAutoEncoder
    :type dataset: RewardModelDataset

    :param model:
    :param dataset:
    :param seed_to_optimize:
    :param mode:
    :param init:
    :param max_iters:
    :param iters_per_print:
    :param gradient_max_norm:
    :param rng_seed:
    :param verbose:
    :return:
    """
    if max_iters <= 0:
        raise ValueError("max_iters must be strictly positive")

    if type(seed_to_optimize) != torch.tensor:
        seed_to_optimize = torch.tensor(seed_to_optimize)

    if verbose > 0:
        print(f"Optimizing reward (seed = {seed_to_optimize}, mode = {mode}, init = {init})...")

    model.eval()
    if verbose > 0:
        print("Note: model has been set to eval mode")

    rng = torch.Generator()
    rng.manual_seed(rng_seed)

    if init == "best_of_dataset":
        best_individual, best_fitness = None, float('inf')

        for ind, seed, fitness in dataset:
            if seed != seed_to_optimize:
                continue

            if fitness < best_fitness:
                best_individual = ind
                best_fitness = fitness

        x = best_individual
    elif init == "zeros":
        x = torch.zeros(size=(model.input_size.item(),))
    elif init == "random":
        x = -1. + 2. * torch.rand(size=(model.input_size.item(),), generator=rng)
    else:
        raise NotImplementedError(f"init '{init}' not implemented")

    optimizer, scheduler = None, None

    def sa_neighbor(y):
        with torch.no_grad():
            result = y.clone()
            k = torch.randint(size=(1,), low=0, high=y.shape[0], generator=rng)
            result[k] = -1. + 2. * torch.rand(size=(1,), generator=rng)

        return result

    def sa_temperature(i):
        return 1. - i / max_iters

    def sa_energy(y):
        with torch.no_grad():
            result = model(y, seed_to_optimize)

        return result

    def sa_acceptance_prob(energy_x, energy_y, T):
        return torch.exp(-torch.abs((energy_x - energy_y) / T)).clamp_(max=1.)

    if mode == "adam":
        x.requires_grad_()

        optimizer = optim.Adam(params=[x], lr=0.001)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1. / (1. + 0.01 * epoch))
    elif mode == "sa":
        pass
    else:
        raise NotImplementedError(f"mode '{mode}' not implemented in the optimizer initialization step")

    fitness = float('nan')
    for iteration in range(1, max_iters + 1):
        if mode == "adam":
            fitness = model(x, seed_to_optimize)

            fitness.backward()

            grad_norm = nn.utils.clip_grad_norm_(x,
                                                 max_norm=gradient_max_norm,
                                                 error_if_nonfinite=True)

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                x.clamp_(min=-1, max=1)
        elif mode == "sa":
            T = sa_temperature(iteration)
            y = sa_neighbor(x)
            fitness = sa_energy(x)
            p = sa_acceptance_prob(fitness, sa_energy(y), T)

            if torch.rand(size=(1,), generator=rng) < p:
                x = y
        else:
            raise NotImplementedError(f"mode '{mode}' not implemented in the optimization step")

        if verbose > 0 and (iteration % iters_per_print == 0 or iteration == 1 or iteration == max_iters):
            print(
                f"Iteration {str(iteration).ljust(6)}: Fitness = {fitness.item():.3f}, Individual = {string_from_onehots(autoencoder.decoder(x))}")
    if verbose > 0:
        print(f"Final result:\n{x}")

    return x
