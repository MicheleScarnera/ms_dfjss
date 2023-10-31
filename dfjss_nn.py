import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import pandas as pd

import time
import datetime
import os

import dfjss_priorityfunction as pf
import dfjss_genetic as genetic
import dfjss_misc as misc

torch.autograd.set_detect_anomaly(True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device.upper()}")

INDIVIDUALS_FILENAME = "individuals.csv"
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


class RNNEncoderHat(nn.Module):
    module: nn.Module

    def __init__(self, rnn, embedding):
        super().__init__()
        self.rnn = rnn
        self.embedding = embedding
        # self.layer_norm = nn.LayerNorm(rnn.hidden_size)

    def forward(self, x, h):
        is_batch = len(x.shape) == 3

        if is_batch:
            x_reduced_info = reduce_sequence(x)[:, :, FIRST_NONFUNCTIONAL_REDUCED_TOKEN_INDEX:len(VOCAB_REDUCED)]
        else:
            x_reduced_info = reduce_sequence(x)[:, FIRST_NONFUNCTIONAL_REDUCED_TOKEN_INDEX:len(VOCAB_REDUCED)]

        argmax_dim = 2 if is_batch else 1
        e = self.embedding(torch.argmax(x, dim=argmax_dim))

        rnn_out = self.rnn.forward(torch.cat([e, x_reduced_info], dim=argmax_dim), h)
        return rnn_out[0], rnn_out[1]


class RNNDecoderHat(nn.Module):
    rnn: nn.RNN

    def __init__(self, rnn, actual_input_size):
        super().__init__()
        self.rnn = rnn
        self.actual_input_size = actual_input_size
        self.proj = nn.Linear(in_features=rnn.hidden_size, out_features=actual_input_size, bias=False)
        # self.layer_norm = nn.LayerNorm(rnn.hidden_size)

    def forward(self, x, h, reverse_input_h=True):
        if reverse_input_h:
            h_ = torch.flip(h, dims=[0])
        else:
            h_ = h

        rnn_out = self.rnn.forward(x, h_)

        # return self.proj(rnn_out[0]), rnn_out[1]

        if len(x.shape) == 2:
            decoder_y = rnn_out[0]  # self.layer_norm(rnn_out[0])
            output = nn.functional.log_softmax(self.proj(decoder_y), dim=1)

            return decoder_y, rnn_out[1], output
        elif len(x.shape) == 3:
            decoder_y = rnn_out[0]  # self.layer_norm(rnn_out[0])
            output = nn.functional.log_softmax(self.proj(decoder_y), dim=2)

            return decoder_y, rnn_out[1], output
        else:
            raise ValueError(f"Expected either 2-D or 3-D, got {len(x.shape)}-D")


def new_first_decoder_token(confidence=1, dtype=torch.float32):
    result = np.full(shape=VOCAB_SIZE, fill_value=0)
    result[VOCAB.index("(")] = confidence
    return torch.tensor(result, dtype=dtype)


class IndividualRNNAutoEncoder(nn.Module):
    def __init__(self, input_size=None, hidden_size=512, num_layers=3, dropout=0.1, bidirectional=False,
                 nonlinearity='tanh', embedding_dim=64):
        super(IndividualRNNAutoEncoder, self).__init__()

        if input_size is None:
            input_size = len(VOCAB)

        self.input_size = input_size

        input_size_encoder = embedding_dim + len(
            VOCAB_REDUCED[FIRST_NONFUNCTIONAL_REDUCED_TOKEN_INDEX:len(VOCAB_REDUCED)])

        self.d = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.encoder_embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)

        self.encoder = RNNEncoderHat(nn.RNN(input_size=input_size_encoder,
                                            hidden_size=hidden_size,
                                            num_layers=num_layers,
                                            dropout=dropout,
                                            bidirectional=bidirectional,
                                            batch_first=True,
                                            nonlinearity=nonlinearity,
                                            device=device), self.encoder_embedding)

        self.encoder_output_size = self.d * hidden_size

        self.decoder = RNNDecoderHat(nn.RNN(input_size=hidden_size,
                                            hidden_size=hidden_size,
                                            num_layers=num_layers,
                                            dropout=dropout,
                                            bidirectional=False,
                                            batch_first=False,
                                            nonlinearity=nonlinearity,
                                            device=device), input_size)

        # for p in self.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -gradient_clip_value, gradient_clip_value))

    def count_parameters(self, learnable=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad == learnable)

    def summary(self):
        longdash = '------------------------------------'
        result = [longdash, "Individual RNN Auto-Encoder",
                  f"Vocabulary size: {self.input_size} (Embedding dimensions: {self.encoder_embedding.embedding_dim}, {self.encoder.rnn.input_size} including o/F indicators)",
                  f"Hidden size: {self.encoder.rnn.hidden_size}", f"Layers: {self.encoder.rnn.num_layers}",
                  f"Dropout: {self.encoder.rnn.dropout}", f"Bidirectional: {self.encoder.rnn.bidirectional}",
                  f"Number of parameters: (Learnable: {self.count_parameters()}, Fixed: {self.count_parameters(False)})",
                  longdash]

        return "\n".join(result)

    def forward(self, x):
        is_batch = len(x.shape) == 3
        batch_size = x.shape[0] if is_batch else None

        sequence_length = x.shape[1] if is_batch else x.shape[0]

        encoder_h_size = (self.d * self.num_layers, batch_size, self.encoder.rnn.hidden_size) if is_batch else (
            self.d * self.num_layers, self.encoder.rnn.hidden_size)
        decoder_h_size = (self.num_layers, batch_size, self.decoder.rnn.hidden_size) if is_batch else (
            self.num_layers, self.decoder.rnn.hidden_size)

        decoder_x_size = (1, batch_size, self.decoder.rnn.input_size) if is_batch else (
            (1, self.decoder.rnn.input_size))

        encoder_y, encoder_h = self.encoder(x, torch.zeros(encoder_h_size))

        decoder_x = torch.zeros(
            size=decoder_x_size)  # torch.transpose(encoder_y[:, -1, :].unsqueeze(1), 0, 1) if is_batch else encoder_y[-1, :].unsqueeze(0)
        current_decoder_h = encoder_h  # torch.zeros(decoder_h_size)

        decodes = []
        for _ in range(sequence_length):
            decoder_x, current_decoder_h, d = self.decoder(decoder_x, current_decoder_h)

            decodes.append(d)

        return torch.transpose(torch.cat(decodes, dim=0), 0, 1) if is_batch else torch.cat(decodes, dim=0)


class IndividualTransformerAutoEncoder(nn.Module):
    def __init__(self, nhead=9, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512,
                 dropout=0.1, max_length=34):
        super(IndividualTransformerAutoEncoder, self).__init__()

        # self.embedding = nn.Embedding(VOCAB_SIZE, d_model)

        self.transformer = nn.Transformer(d_model=VOCAB_SIZE,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True,
                                          device=device)

        self.dropout = dropout
        self.max_length = max_length

    def auto_encode(self, sequence):
        # embedded = self.embedding(index_sequence)
        return nn.functional.log_softmax(self.transformer(sequence, sequence), dim=-1)

    def encode(self, sequence):
        return self.transformer.encoder(sequence)

    def decode(self, encoded, target):
        return nn.functional.log_softmax(self.transformer(target, encoded), dim=-1)

    def blind_decode(self, encoded):
        if encoded.dim() == 3:
            return [self.blind_decode(e) for e in encoded]

        # Initialize the generated sequence with the start token
        generated_sequence_sparse = [VOCAB.index("(")]
        generated_sequence = [nn.functional.log_softmax(new_first_decoder_token(8), dim=0)]
        end_token = VOCAB.index("EOS")

        with torch.no_grad():
            for _ in range(self.max_length - 1):
                current_decode = torch.stack(generated_sequence, dim=0).to(device)

                output = self.transformer.decoder(current_decode, encoded)

                generated_sequence.append(nn.functional.log_softmax(output[-1, :], dim=-1))

                predicted_token_sparse = output[-1, :].argmax().item()
                generated_sequence_sparse.append(predicted_token_sparse)

                # If the end token is predicted, stop decoding
                if predicted_token_sparse == end_token:
                    break

        result_decode = torch.stack(generated_sequence, dim=0).to(device)

        return result_decode, torch.tensor(generated_sequence_sparse)

    def blind_auto_encode(self, sequence):
        return self.blind_decode(self.encode(sequence))

    def forward(self, x):
        return self.auto_encode(x)

    def count_parameters(self, learnable=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad == learnable)

    def summary(self):
        longdash = '------------------------------------'
        result = [longdash, "Individual Transformer Auto-Encoder",
                  f"Vocabulary size: {self.transformer.d_model}",
                  f"Max Length: {self.max_length}",
                  f"Heads: {self.transformer.nhead}",
                  f"Layers: {self.transformer.encoder.num_layers} (Encoder), {self.transformer.decoder.num_layers} (Decoder)",
                  f"Dropout: {self.dropout}",
                  f"Number of parameters: (Learnable: {self.count_parameters()}, Fixed: {self.count_parameters(False)})",
                  longdash]

        return "\n".join(result)


class IndividualFeedForwardAutoEncoder(nn.Module):
    def __init__(self, sequence_length, embedding_dim=-1, hidden_size=2400, encoding_size=1200, dropout=0.1):
        super(IndividualFeedForwardAutoEncoder, self).__init__()

        if embedding_dim > 0:
            self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE,
                                          embedding_dim=embedding_dim,
                                          device=device)
        else:
            self.embedding = None

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.flat_in_size = sequence_length * (embedding_dim if embedding_dim > 0 else VOCAB_SIZE)
        self.flat_out_size = sequence_length * VOCAB_SIZE

        self.embed_to_flat_in = nn.Flatten(-2, -1)
        self.flat_in_to_h_in = nn.Linear(in_features=self.flat_in_size, out_features=hidden_size, device=device)
        self.h_in_activation = nn.PReLU()

        self.h_in_to_encoder = nn.Linear(in_features=hidden_size, out_features=encoding_size, device=device)
        self.encoder_activation = nn.Softsign()

        self.encoder_dropout = nn.Dropout(p=dropout)

        self.encoder_to_h_out = nn.Linear(in_features=encoding_size, out_features=hidden_size, device=device)
        self.h_out_activation = nn.PReLU()
        self.h_out_to_flat_out = nn.Linear(in_features=hidden_size, out_features=self.flat_out_size, device=device)
        self.flat_out_activation = nn.PReLU()

        self.flat_out_to_decode = nn.Unflatten(dim=-1, unflattened_size=(sequence_length, VOCAB_SIZE))
        self.decode_activation = nn.LogSoftmax(dim=-1)

    def auto_encoder(self, sequence):
        return self.decoder(self.encoder(sequence))

    def encoder(self, sequence):
        if torch.is_floating_point(sequence):
            indices = torch.argmax(sequence, dim=-1)
        else:
            indices = sequence

        embed = self.embedding(indices) if self.embedding is not None else sequence
        flattened = self.embed_to_flat_in(embed)

        h_in = self.h_in_activation(self.flat_in_to_h_in(flattened))

        return self.encoder_activation(self.h_in_to_encoder(h_in))

    def decoder(self, encoding):
        h_out = self.h_out_activation(self.encoder_to_h_out(self.encoder_dropout(encoding)))
        flat_out = self.flat_out_activation(self.h_out_to_flat_out(h_out))

        return self.decode_activation(self.flat_out_to_decode(flat_out))

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
                  f"Hidden Size: {self.hidden_size}", f"Encoding Size: {self.encoding_size}",
                  f"Encoder Dropout: {self.encoder_dropout.p}",
                  f"Number of parameters: {self.count_parameters()} (Learnable), {self.count_parameters(False)} (Fixed)",
                  longdash]

        return "\n".join(result)


def generate_individuals_file(total_amount=20000, max_depth=8, rng_seed=100):
    gen_algo = genetic.GeneticAlgorithm(rng_seed=rng_seed)
    gen_algo.settings.features = INDIVIDUALS_FEATURES

    D = (max_depth - 2 + 1)

    amount_per_depth = [total_amount // D for _ in range(2, max_depth + 1)]

    for j in range(total_amount - sum(amount_per_depth)):
        amount_per_depth[j % (max_depth - 2 + 1)] += 1

    df = pd.DataFrame(columns=["Individual"])
    start = time.time()

    current_amount = 1
    for d_ in range(D):
        d = d_ + 2
        gen_algo.settings.tree_generation_max_depth = d
        for _ in range(amount_per_depth[d_]):
            df.loc[len(df)] = {"Individual": repr(gen_algo.get_random_individual())}

            print(
                f"\rGenerating individuals... {current_amount / total_amount:.1%} {misc.timeformat(misc.timeleft(start, time.time(), current_amount, total_amount))}",
                end="", flush=True)

            current_amount += 1

    df.to_csv(path_or_buf=INDIVIDUALS_FILENAME, index=False)

    print(f"\rTook {misc.timeformat(time.time() - start)}", flush=True)

    return df


def one_hot_sequence(individual, vocab=None):
    return torch.stack([token_to_one_hot(s, vocab=vocab) for s in tokenize_with_vocab(individual, vocab=vocab)])


def sparse_sequence(individual, vocab=None):
    return torch.stack([token_to_sparse(s, vocab=vocab) for s in tokenize_with_vocab(individual, vocab=vocab)])


def reduce_sequence(sequence, input_is_logs=False):
    if len(sequence.shape) == 3:
        return torch.stack([reduce_sequence(x_) for x_ in sequence], dim=0)

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
    def __init__(self, rng_seed=100, max_depth=4, size=5000, refresh_rate=0., fill_trees=True, flatten_trees=True,
                 sparse=False,
                 refresh_is_random=False):
        super().__init__()

        self.gen_algo = genetic.GeneticAlgorithm(rng_seed=rng_seed)
        self.gen_algo.settings.features = INDIVIDUALS_FEATURES
        self.gen_algo.settings.tree_generation_max_depth = max_depth
        self.gen_algo.settings.tree_generation_fill = fill_trees

        self.max_depth = max_depth
        self.fill_trees = fill_trees
        self.flatten_trees = flatten_trees
        self.size = size
        self.refresh_rate = refresh_rate
        self.refresh_is_random = refresh_is_random
        self.sparse = sparse

        self.individuals = []
        self.times_refreshed = 0
        self.total_datapoints = 0

        self.rng = np.random.default_rng(rng_seed)

        self.fill_individuals()

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
                flat[i] = self.rng.choice(a=FEATURES_AND_CONSTANTS)

        tree.set_from_flattened(flat)

        return tree

    def fill_individuals(self):
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

            del individual

            self.individuals.append(sparse_sequence(representation).to(device) if self.sparse else (
                one_hot_sequence(representation).to(device, torch.float32)))

            self.total_datapoints += 1

    def refresh_data(self):
        # permanence rate
        if self.refresh_rate > 0.:
            for _ in range(int(self.size * self.refresh_rate)):
                index = self.rng.choice(len(self.individuals)) if self.refresh_is_random else 0

                self.individuals.pop(index)

        # if length less than size, generate individuals
        self.fill_individuals()

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


def sequence_collate(batch):
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
                      batch_size=16,
                      max_depth=8,
                      flatten_trees=True,
                      fill_trees=True,
                      train_size=5000,
                      train_refresh_rate=1.,
                      train_seed=100,
                      val_size=1000,
                      val_refresh_rate=0.,
                      val_seed=1337,
                      raw_criterion_weight=0.3,
                      raw_criterion_weight_inc=0.05,
                      reduced_criterion_weight=0.7,
                      reduced_criterion_weight_inc=0.,
                      feature_classes_weight=4,
                      operation_classes_weight=2,
                      other_classes_weight=1,
                      syntax_score_coefficient=0.,
                      gradient_value_threshold=1.,
                      clipping_is_norm=True,
                      gradient_norm_threshold=1.,
                      clipping_norm_type=2.):
    """
    Train an autoencoder.

    :type model: nn.Module

    :param model: The autoencoder model.
    :param max_depth: Individuals will not exceed this depth.
    :param flatten_trees: Whether data should be of flat individuals or not. You cannot have flat individuals if they are not also full.
    :param fill_trees: Whether individuals should be full. You cannot have flat individuals if they are not also full.
    :param num_epochs: The maximum number of epochs to train on.
    :param train_size: The size of the training set each epoch.
    :param train_refresh_rate: How much, in percentage, the training set is replaced with new data each epoch.
    :param train_seed: The seed for the training data.
    :param val_size: The size of the validation set each epoch. If the training set has non-zero refresh rate, it makes sense for this to be large, even larger than train_size.
    :param val_refresh_rate: How much, in percentage, the validation set is replaced with new data each epoch. It is recommended to keep this to 0 for consistency of evaluation.
    :param val_seed: The seed for the validation data.
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
        raise ValueError("You cannot have flat individuals if they are not also full. Change the flatten_trees and fill_trees arguments")

    model_type = type(model)

    if model_type == IndividualRNNAutoEncoder:
        autoencoder_type = "RNN"
    elif model_type == IndividualTransformerAutoEncoder:
        autoencoder_type = "TRANSFORMER"
    elif model_type == IndividualFeedForwardAutoEncoder:
        autoencoder_type = "FEEDFORWARD"
    else:
        raise TypeError(f"Unknown autoencoder type ({model_type})")

    if autoencoder_type == "FEEDFORWARD" and not fill_trees:
        raise ValueError("Feed-Forward auto-encoders can only be trained with full trees, as they require the input to be of constant length. Set fill_trees to True")

    folder_name = datetime.datetime.now().strftime(f'AUTOENCODER {autoencoder_type} %Y-%m-%d %H-%M-%S')

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print(f"Model(s) will be saved in \"{folder_name}\"")

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

    if autoencoder_type == "TRANSFORMER":
        df_cols.append("BlindAutoencodedExample")

    df = pd.DataFrame(columns=df_cols)

    df_examples = pd.DataFrame(columns=df_examples_cols)

    train_set = AutoencoderDataset(max_depth=max_depth,
                                   size=train_size,
                                   refresh_rate=train_refresh_rate,
                                   flatten_trees=flatten_trees,
                                   fill_trees=fill_trees,
                                   rng_seed=train_seed)

    val_set = AutoencoderDataset(max_depth=max_depth,
                                 size=val_size,
                                 refresh_rate=val_refresh_rate,
                                 flatten_trees=flatten_trees,
                                 fill_trees=fill_trees,
                                 rng_seed=val_seed)

    implied_length = train_set.max_sequence_length()
    if autoencoder_type == "FEEDFORWARD" and model.sequence_length != implied_length:
        raise ValueError(
            f"The feed-forward network needs individuals whose length is exactly {model.sequence_length}, but the max_depth ({max_depth}) parameter implies a length of {implied_length}.")

    print(f"Creating training and validation set for Epoch 1...", end="")
    
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=sequence_collate,
        shuffle=True
    )

    val_loader = data.DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=sequence_collate,
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

    raw_criterion_class_weights = torch.tensor([class_weight(token) for token in VOCAB], dtype=torch.float32)
    criterion_raw = nn.NLLLoss(weight=raw_criterion_class_weights)
    criterion_reduced = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # optim.SGD(model.parameters(), lr=0.001, momentum=0.25)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max", threshold=0.005, patience=2, verbose=False)

    reduced_criterion_scale = 1.  # np.log(len(VOCAB)) / np.log(len(VOCAB_REDUCED))

    for epoch in range(1, num_epochs + 1):
        # Training
        train_loss = 0.
        train_raw_criterion = 0.
        train_reduced_criterion = 0.
        train_total_criterion = 0.
        train_largest_criterion = 0.
        train_syntaxscore = 0.

        train_accuracy = 0.
        train_valid = 0.
        train_perfect_matches = 0

        train_start = None
        train_progress = 0
        train_progress_needed = None

        # Validation

        val_loss = 0.
        val_raw_criterion = 0.
        val_reduced_criterion = 0.
        val_total_criterion = 0.
        val_syntaxscore = 0.

        val_accuracy = 0.
        val_valid = 0.
        val_perfect_matches = 0

        val_example = ""
        val_autoencoded = ""
        val_blind_autoencoded = ""

        val_examples = []

        val_start = None
        val_progress = 0
        val_progress_needed = None

        # Raw/Reduced criterion weights
        criterion_weights_raw = (epoch - 1) * raw_criterion_weight_inc + raw_criterion_weight
        criterion_weights_reduced = (epoch - 1) * reduced_criterion_weight_inc + reduced_criterion_weight
        criterion_weights_norm = criterion_weights_raw + criterion_weights_reduced
        criterion_weights_raw /= criterion_weights_norm
        criterion_weights_reduced /= criterion_weights_norm

        for is_train in (True, False):
            gradients_thrown_out = None
            if is_train:
                print(f"\rEpoch {epoch}: Training...", end="")
                model.train()

                train_start = time.time()

                gradients_thrown_out = []
            else:
                print(f"\rEpoch {epoch}: Validating...", end="", flush=True)
                model.eval()

                val_start = time.time()

            loader = train_loader if is_train else val_loader

            for true_sequences in loader:
                losses = None

                if is_train:
                    optimizer.zero_grad()
                    losses = []

                true_sequences = true_sequences.to(device)
                true_sequences_sparse = torch.argmax(true_sequences, dim=2)

                outputs = model(true_sequences).to(device)
                B = true_sequences.shape[0]
                for b in range(B):
                    if is_train:
                        if train_progress_needed is None:
                            train_progress_needed = len(train_loader) * B
                    else:
                        if val_progress_needed is None:
                            val_progress_needed = len(val_loader) * B

                    true_tokens = string_from_onehots(true_sequences[b], list_mode=True)
                    output_tokens = string_from_onehots(outputs[b], list_mode=True)

                    eos_cutoff = true_tokens.index("EOS")

                    # +1 because EOS needs to be present
                    eos_cutoff = min(eos_cutoff + 1, len(true_tokens))

                    true_tokens = true_tokens[0:eos_cutoff]
                    output_tokens = output_tokens[0:eos_cutoff]

                    output = outputs[b, 0:eos_cutoff, :]
                    true_sequence = true_sequences[b, 0:eos_cutoff, :]
                    true_sequence_sparse = true_sequences_sparse[b, 0:eos_cutoff]

                    output_reduced = reduce_sequence(output, input_is_logs=True)
                    true_sequence_reduced = reduce_sequence(true_sequence)
                    true_sequence_reduced_sparse = torch.argmax(true_sequence_reduced, dim=1)

                    if not is_train:
                        list_true = string_from_onehots(true_sequence, list_mode=True)
                        list_output = string_from_onehots(output, list_mode=True)
                        str_true = "".join(list_true)
                        str_output = "".join(list_output)

                        if val_example == "":
                            val_example = str_true
                            val_autoencoded = str_output

                            if autoencoder_type == "TRANSFORMER":
                                val_blind_autoencoded = string_from_sparse(model.blind_auto_encode(true_sequence)[1])

                        new_example = dict()
                        new_example["Epoch"] = epoch
                        new_example["Example"] = str_true
                        new_example["AutoencodedExample"] = str_output
                        new_example["Accuracy"] = np.mean([t == o for t, o in zip(list_true, list_output)])

                        val_examples.append(new_example)

                    loss_raw_criterion = criterion_raw(output, true_sequence_sparse)
                    loss_reduced_criterion = reduced_criterion_scale * criterion_reduced(output_reduced,
                                                                                         true_sequence_reduced_sparse)

                    if loss_reduced_criterion < 0.0005:
                        loss_criterion = loss_raw_criterion
                    else:
                        loss_criterion = criterion_weights_raw * loss_raw_criterion + (
                                criterion_weights_reduced * loss_reduced_criterion)

                    if flatten_trees or syntax_score_coefficient <= 0.:
                        loss_syntax_penalty = torch.tensor([0.])
                        sntx = torch.tensor([0.])
                    else:
                        loss_syntax_penalty, sntx = syntax_penalty_term_and_syntax_score(output[0:-1, :],
                                                                                         lam=syntax_score_coefficient)
                    loss = loss_criterion + loss_syntax_penalty

                    if is_train:
                        if torch.isfinite(loss):
                            losses.append(loss)
                        else:
                            print(f"WARNING: a non-finite loss was encountered ({loss}). It will not be considered")

                        train_loss += loss.item()
                        train_raw_criterion += loss_raw_criterion.item()
                        train_reduced_criterion += loss_reduced_criterion.item()
                        train_total_criterion += loss_criterion.item()
                        train_syntaxscore += sntx.item()

                        if loss_criterion.item() > train_largest_criterion:
                            train_largest_criterion = loss_criterion.item()
                    else:
                        val_loss += loss.item()
                        val_raw_criterion += loss_raw_criterion.item()
                        val_reduced_criterion += loss_reduced_criterion.item()
                        val_total_criterion += loss_criterion.item()
                        val_syntaxscore += sntx.item()

                    found_mismatch = False

                    T = len(true_tokens)
                    matches = 0
                    for t in range(T):
                        true_token = true_tokens[t]
                        output_token = output_tokens[t]

                        if true_token == output_token:
                            matches += 1
                        else:
                            found_mismatch = True

                        if t == (T - 1):
                            if is_train:
                                train_accuracy += matches / T
                            else:
                                val_accuracy += matches / T
                            break

                    if not found_mismatch:
                        if is_train:
                            train_perfect_matches += 1
                        else:
                            val_perfect_matches += 1

                    if not flatten_trees:
                        if pf.is_representation_valid("".join(output_tokens[0:-1]), features=INDIVIDUALS_FEATURES):
                            if is_train:
                                train_valid += 1. / B
                            else:
                                val_valid += 1. / B

                    if is_train:
                        train_progress += 1

                        datapoints_per_second = train_progress / (time.time() - train_start)
                    else:
                        val_progress += 1

                        datapoints_per_second = val_progress / (time.time() - val_start)

                    dps_text = f"{datapoints_per_second:.2f} datapoints per second" if datapoints_per_second > 1. else f"{misc.timeformat(1. / datapoints_per_second)} per datapoint"

                    if is_train:
                        gradient_norms_text = f"{np.mean(gradients_thrown_out):.2%}" if len(
                            gradients_thrown_out) > 0 else "N/A"
                        print(
                            f"\rEpoch {epoch}: Training... {train_progress / train_progress_needed:.1%} ETA {misc.timeformat(misc.timeleft(train_start, time.time(), train_progress, train_progress_needed))}, {dps_text} (Average criterion: {train_total_criterion / train_progress:.3f} ({criterion_weights_raw:.2f}*{train_raw_criterion / train_progress:.3f}+{criterion_weights_reduced:.2f}*{train_reduced_criterion / train_progress:.3f}), Largest criterion: {train_largest_criterion:.3f}, Gradients' norms lost due to clipping: {gradient_norms_text})",
                            end="", flush=True)
                    else:
                        print(
                            f"\rEpoch {epoch}: Validating... {val_progress / val_progress_needed:.1%} ETA {misc.timeformat(misc.timeleft(val_start, time.time(), val_progress, val_progress_needed))}, {dps_text}",
                            end="", flush=True)

                if is_train:
                    losses = torch.stack(losses)
                    losses.mean().backward()

                    # zero out NaN gradients
                    for p in model.parameters():
                        p.grad[torch.isnan(p.grad)] = 0

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
                            np.mean(np.max([(grads - gradient_value_threshold) / grads, np.zeros(len(grads))], axis=0,
                                           where=grads > 0., initial=0.)))

                        nn.utils.clip_grad_value_(model.parameters(), gradient_value_threshold)

                    optimizer.step()

        l_t = train_progress
        l_v = val_progress

        train_loss /= l_t
        train_raw_criterion /= l_t
        train_reduced_criterion /= l_t
        train_total_criterion /= l_t
        train_syntaxscore /= l_t

        train_accuracy /= l_t
        train_valid /= l_t
        train_perfect_matches /= l_t

        de_facto_raw_weight = (train_total_criterion - train_reduced_criterion) / (
                train_raw_criterion - train_reduced_criterion) if train_raw_criterion > train_reduced_criterion else 1.
        de_facto_reduced_weight = 1. - de_facto_raw_weight

        val_loss /= l_v
        val_raw_criterion /= l_v
        val_reduced_criterion /= l_v
        val_total_criterion = de_facto_raw_weight * val_raw_criterion + de_facto_reduced_weight * val_reduced_criterion
        val_syntaxscore /= l_v

        val_accuracy /= l_v
        val_valid /= l_v
        val_perfect_matches /= l_v

        current_lr = optimizer.param_groups[0]['lr']

        if val_perfect_matches > 0.:
            scheduler.step(val_perfect_matches)

        print(
            f"\rEpoch {epoch}: Loss/Criterion/Syntax/Valid/Accuracy/Perfects: (Train: {train_loss:.4f}/{train_total_criterion:.4f} ({de_facto_raw_weight:.2f}*{train_raw_criterion:.3f}+{de_facto_reduced_weight:.2f}*{train_reduced_criterion:.3f})/{train_syntaxscore:.2%}/{train_valid:.2%}/{train_accuracy:.2%}/{train_perfect_matches:.2%}) (Val: {val_loss:.4f}/{val_total_criterion:.4f} ({de_facto_raw_weight:.2f}*{val_raw_criterion:.3f}+{de_facto_reduced_weight:.2f}*{val_reduced_criterion:.3f})/{val_syntaxscore:.2%}/{val_valid:.2%}/{val_accuracy:.2%}/{val_perfect_matches:.2%}) (Total data: {train_set.total_datapoints}, {val_set.total_datapoints}) LR: {current_lr:.0e} Took {misc.timeformat(time.time() - train_start)} ({misc.timeformat(val_start - train_start)}, {misc.timeformat(time.time() - val_start)})"
        )

        new_row = dict()
        new_row["Epoch"] = epoch
        new_row["Criterion_Weight_Raw"] = de_facto_raw_weight
        new_row["Criterion_Weight_Reduced"] = de_facto_reduced_weight
        new_row["Train_TotalDatapoints"] = train_set.total_datapoints
        new_row["Train_Loss"] = train_loss
        new_row["Train_Total_Criterion"] = train_total_criterion
        new_row["Train_Raw_Criterion"] = train_raw_criterion
        new_row["Train_Reduced_Criterion"] = train_reduced_criterion
        new_row["Train_SyntaxScore"] = train_syntaxscore
        new_row["Train_Valid"] = train_valid
        new_row["Train_Accuracy"] = train_accuracy
        new_row["Train_Perfects"] = train_perfect_matches
        new_row["Train_LR"] = current_lr

        new_row["Val_TotalDatapoints"] = val_set.total_datapoints
        new_row["Val_Loss"] = val_loss
        new_row["Val_Total_Criterion"] = val_total_criterion
        new_row["Val_Raw_Criterion"] = val_raw_criterion
        new_row["Val_Reduced_Criterion"] = val_reduced_criterion
        new_row["Val_SyntaxScore"] = val_syntaxscore
        new_row["Val_Valid"] = val_valid
        new_row["Val_Accuracy"] = val_accuracy
        new_row["Val_Perfects"] = val_perfect_matches

        new_row["Example"] = val_example
        new_row["AutoencodedExample"] = val_autoencoded

        if len(df) > 0:
            df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        else:
            df = pd.DataFrame(new_row, index=[0])

        df.to_csv(path_or_buf=f"{folder_name}/log.csv", index=False)

        if len(df_examples) > 0:
            df_examples = pd.concat([df_examples, pd.DataFrame(val_examples, index=range(len(val_examples)))],
                                    ignore_index=True)
        else:
            df_examples = pd.DataFrame(val_examples, index=range(len(val_examples)))

        df_examples.sort_values(by=["Example", "Epoch"], inplace=True)

        df_examples.to_csv(path_or_buf=f"{folder_name}/examples.csv", index=False)

        torch.save(model.state_dict(), f"{folder_name}/model_epoch{epoch}.pth")

        if val_perfect_matches >= 1.:
            print("The model has reached 100% perfect matches in the validation set, and has stopped training")
            break

        if epoch < num_epochs:
            print(f"Refreshing training and validation set for Epoch {epoch+1}...", end="")
            train_set.refresh_data()
            val_set.refresh_data()

