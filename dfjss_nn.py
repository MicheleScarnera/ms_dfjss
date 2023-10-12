import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import pandas as pd

import time
import datetime
import os

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

VOCAB = ["NULL", "EOS", "(", ")"]  # , *OPERATIONS, *INDIVIDUALS_FEATURES, *CONSTANTS
VOCAB_REDUCED = VOCAB.copy()

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


class EncoderHat(nn.Module):
    module: nn.Module

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    def forward(self, x, h):
        return torch.atanh(self.rnn.forward(x, h)[1][-1, :].unsqueeze_(0))


class DecoderTail(nn.Module):
    rnn: nn.Module

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    def forward(self, x, h):
        return self.rnn.forward(torch.softmax(x, dim=1), h)


class IndividualAutoEncoder(nn.Module):
    def __init__(self, input_size=None, hidden_size=512, num_layers=2, dropout=0.5, bidirectional=False):
        super(IndividualAutoEncoder, self).__init__()

        if input_size is None:
            input_size = len(VOCAB)

        self.d = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.encoder = EncoderHat(nn.RNN(input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=num_layers,
                                         dropout=dropout,
                                         bidirectional=bidirectional,
                                         batch_first=True,
                                         device=device))

        self.encoder_output_size = self.d * hidden_size

        self.decoder = DecoderTail(nn.RNN(input_size=self.encoder_output_size,
                                          hidden_size=input_size,
                                          num_layers=num_layers,
                                          dropout=dropout,
                                          bidirectional=False,
                                          batch_first=False,
                                          device=device))

    def count_parameters(self, learnable=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad == learnable)

    def summary(self):
        longdash = '------------------------------------'
        result = [longdash, "Individual Auto-Encoder", f"Input size: {self.encoder.rnn.input_size}",
                  f"Hidden size: {self.encoder.rnn.hidden_size}", f"Layers: {self.encoder.rnn.num_layers}",
                  f"Dropout: {self.encoder.rnn.dropout}", f"Bidirectional: {self.encoder.rnn.bidirectional}",
                  f"Number of parameters: (Learnable: {self.count_parameters()}, Fixed: {self.count_parameters(False)})",
                  longdash]

        return "\n".join(result)

    def forward(self, x):
        is_batch = len(x.shape) == 3
        batch_size = x.shape[0] if is_batch else None

        vocab_length = self.encoder.rnn.input_size
        sequence_length = x.shape[1] if is_batch else x.shape[0]

        encoder_h_size = (self.d * self.num_layers, batch_size, self.encoder.rnn.hidden_size) if is_batch else (
            self.d * self.num_layers, self.encoder.rnn.hidden_size)
        decoder_h_size = (self.num_layers, batch_size, self.decoder.rnn.hidden_size) if is_batch else (
            self.num_layers, self.decoder.rnn.hidden_size)

        encoded = self.encoder(x, torch.zeros(encoder_h_size))

        current_decoder_h = torch.zeros(decoder_h_size)

        decodes = []
        for _ in range(sequence_length):
            d, current_decoder_h = self.decoder(encoded, current_decoder_h)

            decodes.append(d[-1, :])

        return torch.transpose(torch.stack(decodes), 0, 1) if is_batch else torch.stack(decodes)


def generate_individuals_file(total_amount=2500, max_depth=8, rng_seed=100):
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


def reduce_sequence(sequence):
    if sequence.shape[1] == VOCAB_REDUCED_SIZE:
        raise ValueError("Sequence is already reduced")

    if sequence.shape[1] != VOCAB_SIZE:
        raise ValueError(
            f"Sequence is of unexpected vocabulary (expected vocabulary of size {VOCAB_SIZE}, got {sequence.shape[1]})")

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

        """
        if len(possible_words) > 0:
            if vocab == VOCAB and "-" in possible_words and np.any([misc.is_number(pw) for pw in possible_words]) and (len(tokens) > 0 and tokens[-1] not in [*OPERATIONS, "("]):
                possible_words = ["-"]

            max_l = max([len(word) for word in possible_words])

            for word in possible_words:
                if len(word) == max_l:
                    tokens.append(word)
                    i += len(word)
                    matched = True
                    break
        """
        if not matched:
            raise ValueError(f"Could not match character '{input_string[i]}' with anything in the vocabulary {vocab}")

    return tokens


def string_from_onehots(onehots, vocab=None, list_mode=False):
    if len(onehots.shape) > 3:
        raise ValueError(f"Too many dimensions ({len(onehots.shape)})")

    if len(onehots.shape) == 3:
        return torch.cat([string_from_onehots(onehots[i, :], vocab) for i in range(onehots.shape[0])])

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


# "house": sequence of tokens "(FoF)"

HOUSE_DETECTOR_COORDINATES = [(0, 2), (1, 5), (2, 4), (3, 5), (4, 3)]

HOUSE_DETECTOR_FILTER = torch.zeros(size=(5, VOCAB_REDUCED_SIZE))

for i, j in HOUSE_DETECTOR_COORDINATES:
    HOUSE_DETECTOR_FILTER[i, j] = 1.

HOUSE_DETECTOR_MASK = torch.gt(HOUSE_DETECTOR_FILTER, 0.)


def gmean(vector, epsilon=0.01):
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
                                                                                             dim=0).prod()


class IndividualDataset(data.IterableDataset):
    def __init__(self):
        super().__init__()

        # Load the CSV file using pandas
        if os.path.exists(INDIVIDUALS_FILENAME):
            self.df = pd.read_csv(INDIVIDUALS_FILENAME)
        else:
            self.df = generate_individuals_file()

        for i in range(len(self.df)):
            self.df.loc[i, "Individual"] = self.df.loc[i, "Individual"] + "EOS"

    def __iter__(self):
        return self.data_iterator()

    def __getitem__(self, idx):
        return one_hot_sequence(self.df.loc[idx, "Individual"])

    def __len__(self):
        return len(self.df)

    def data_iterator(self):
        # Iterate through each row in the CSV
        for _, row in self.df.iterrows():
            individual = row['Individual']

            yield one_hot_sequence(individual)


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


def regularization_term_and_syntax_score(sequence, lam=10., eps=0.01):
    score = syntax_score(torch.softmax(sequence, dim=1))
    return lam / np.log(eps) * torch.log(score + eps), score


def train_autoencoder(model, dataset, num_epochs=10, batch_size=16, val_split=0.2, regularization_coefficient=10.):
    """
    :type model: nn.Module

    :param model:
    :param dataset:
    :param num_epochs:
    :param val_split:
    :param batch_size:
    :return:
    """
    folder_name = datetime.datetime.now().strftime('AUTOENCODER %Y-%m-%d %H-%M-%S')

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print(f"Model(s) will be saved in \"{folder_name}\"")

    df_cols = ["Epoch"]

    for tv in ("Train", "Val"):
        for q in ("Loss", "Criterion", "SyntaxScore", "Accuracy", "Perfects"):
            df_cols.append(f"{tv}_{q}")

    df = pd.DataFrame(columns=df_cols)

    train_set, val_set = data.random_split(dataset=dataset, lengths=[1. - val_split, val_split])
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max")

    for epoch in range(1, num_epochs + 1):
        # Training
        train_loss = 0.
        train_criterion = 0.
        train_syntaxscore = 0.
        model.train()

        train_accuracy = 0.
        train_perfect_matches = 0

        train_start = time.time()
        train_progress = 0
        train_progress_needed = None

        print(f"Epoch {epoch}: Training...", end="")

        for true_sequences in train_loader:
            true_sequences = true_sequences.to(device)

            optimizer.zero_grad()

            outputs = model(true_sequences).to(device)
            W = true_sequences.shape[0]
            for w in range(W):
                if train_progress_needed is None:
                    train_progress_needed = len(train_loader) * W

                is_last = w == W - 1

                loss_criterion = criterion(outputs[w], true_sequences[w])
                loss_reg, sntx = regularization_term_and_syntax_score(outputs[w], lam=regularization_coefficient)
                loss = loss_criterion + loss_reg
                loss.backward(retain_graph=not is_last)

                train_loss += loss.item()
                train_criterion += loss_criterion.item()
                train_syntaxscore += sntx.item()

                true_tokens = string_from_onehots(true_sequences[w], list_mode=True)
                output_tokens = string_from_onehots(outputs[w], list_mode=True)

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

                    if true_token == "EOS" or t == (T - 1):
                        train_accuracy += matches / (t + 1)
                        break

                if not found_mismatch:
                    train_perfect_matches += 1

                train_progress += 1

                print(
                    f"\rEpoch {epoch}: Training... {train_progress / train_progress_needed:.1%} ETA {misc.timeformat(misc.timeleft(train_start, time.time(), train_progress, train_progress_needed))}",
                    end="", flush=True)

            optimizer.step()

        # Validation
        val_loss = 0.
        val_criterion = 0.
        val_syntaxscore = 0.
        model.eval()

        val_accuracy = 0.
        val_perfect_matches = 0

        val_start = time.time()
        val_progress = 0
        val_progress_needed = None

        print(f"\rEpoch {epoch}: Validating...", end="", flush=True)

        for true_sequences in val_loader:
            true_sequences = true_sequences.to(device)

            outputs = model(true_sequences).to(device)
            for w in range(true_sequences.shape[0]):
                if val_progress_needed is None:
                    val_progress_needed = len(train_loader) * true_sequences.shape[0]

                loss_criterion = criterion(outputs[w], true_sequences[w])
                loss_reg, sntx = regularization_term_and_syntax_score(outputs[w], lam=regularization_coefficient)
                loss = loss_criterion + loss_reg

                val_loss += loss.item()
                val_criterion += loss_criterion.item()
                val_syntaxscore += sntx.item()

                true_tokens = string_from_onehots(true_sequences[w], list_mode=True)
                output_tokens = string_from_onehots(outputs[w], list_mode=True)

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

                    if true_token == "EOS" or t == (T - 1):
                        val_accuracy += matches / (t + 1)
                        break

                if not found_mismatch:
                    val_perfect_matches += 1

                val_progress += 1

                print(
                    f"\rEpoch {epoch}: Validating... {val_progress / val_progress_needed:.1%} ETA {misc.timeformat(misc.timeleft(val_start, time.time(), val_progress, val_progress_needed))}",
                    end="", flush=True)

        scheduler.step(val_accuracy)

        l_t = train_progress
        l_v = val_progress

        train_loss = train_loss / l_t
        train_criterion = train_criterion / l_t
        train_syntaxscore = train_syntaxscore / l_t

        train_accuracy = train_accuracy / l_t
        train_perfect_matches = train_perfect_matches / l_t

        val_loss = val_loss / l_v
        val_criterion = val_criterion / l_v
        val_syntaxscore = val_syntaxscore / l_v

        val_accuracy = val_accuracy / l_v
        val_perfect_matches = val_perfect_matches / l_v

        print(
            f"\rEpoch {epoch}: Loss/Criterion/Syntax/Accuracy/Perfects: (Train: {train_loss:.4f}/{train_criterion:.4f}/{train_syntaxscore:.2%}/{train_accuracy:.2%}/{train_perfect_matches:.2%}) (Val: {val_loss:.4f}/{val_criterion:.4f}/{val_syntaxscore:.2%}/{val_accuracy:.2%}/{val_perfect_matches:.2%}) Took {misc.timeformat(time.time() - train_start)}"
        )

        new_row = dict()
        new_row["Epoch"] = epoch
        new_row["Train_Loss"] = train_loss
        new_row["Train_Criterion"] = train_criterion
        new_row["Train_SyntaxScore"] = train_syntaxscore
        new_row["Train_Accuracy"] = train_accuracy
        new_row["Train_Perfects"] = train_perfect_matches

        new_row["Val_Loss"] = val_loss
        new_row["Val_Criterion"] = val_criterion
        new_row["Val_SyntaxScore"] = val_syntaxscore
        new_row["Val_Accuracy"] = val_accuracy
        new_row["Val_Perfects"] = val_perfect_matches

        if len(df) > 0:
            df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        else:
            df = pd.DataFrame(new_row, index=[0])

        df.to_csv(path_or_buf=f"{folder_name}/log.csv", index=False)

        torch.save(model.state_dict(), f"{folder_name}/model_epoch{epoch}.pth")
