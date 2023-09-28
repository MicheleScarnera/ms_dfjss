import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import pandas as pd

import time
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
VOCAB = ["NULL", "EOS", "(", ")", "+", "-", "*", "/", "<", ">", *INDIVIDUALS_FEATURES]
VOCAB_SIZE = len(VOCAB)

class IndividualAutoEncoder(nn.Module):
    def __init__(self, input_size=None, hidden_size=128, num_layers=1, dropout=0, bidirectional=False):
        super(IndividualAutoEncoder, self).__init__()

        if input_size is None:
            input_size = len(VOCAB)

        self.d = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.encoder = nn.RNN(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              batch_first=True,
                              device=device)

        self.encoder_output_size = self.d * num_layers * hidden_size

        self.decoder = nn.RNN(input_size=self.encoder_output_size,
                              hidden_size=input_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=False,
                              batch_first=False,
                              device=device)

    def forward(self, x):
        is_batch = len(x.shape) == 3
        batch_size = x.shape[0] if is_batch else None

        vocab_length = self.encoder.input_size
        sequence_length = x.shape[1] if is_batch else x.shape[0]

        encoder_h_size = (self.d * self.num_layers, batch_size, self.encoder.hidden_size) if is_batch else (self.d * self.num_layers, self.encoder.hidden_size)
        decoder_h_size = (self.num_layers, batch_size, self.decoder.hidden_size) if is_batch else (self.num_layers, self.decoder.hidden_size)

        _, encoded = self.encoder(x, torch.zeros(encoder_h_size))

        current_decoder_h = torch.zeros(decoder_h_size)

        decodes = []
        for i in range(sequence_length):
            d, current_decoder_h = self.decoder(encoded, current_decoder_h)

            decodes.append(d)

        return torch.transpose(torch.cat(decodes), 0, 1) if is_batch else torch.cat(decodes)


def generate_individuals_file(total_amount=5000, max_depth=8, rng_seed=100):
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

            print(f"\rGenerating individuals... {current_amount / total_amount:.1%} {misc.timeformat(misc.timeleft(start, time.time(), current_amount, total_amount))}", end="", flush=True)

            current_amount += 1

    df.to_csv(path_or_buf=INDIVIDUALS_FILENAME, index=False)

    print(f"\rTook {misc.timeformat(time.time() - start)}", flush=True)

    return df


def one_hot_sequence(individual):
    return torch.stack([token_to_one_hot(s) for s in tokenize_with_vocab(individual)])


def token_to_one_hot(s):
    # Convert a token to a one-hot tensor
    one_hot = torch.zeros(len(VOCAB))
    if s in VOCAB:
        index = VOCAB.index(s)
        one_hot[index] = 1
    else:
        return token_to_one_hot("NULL")
    return one_hot


def tokenize_with_vocab(input_string):
    # Sort the vocab list by length in descending order
    sorted_vocab = sorted(VOCAB, key=lambda x: len(x), reverse=True)

    tokens = []
    i = 0

    while i < len(input_string):
        matched = False

        for word in sorted_vocab:
            if input_string[i:i + len(word)] == word:
                tokens.append(word)
                i += len(word)
                matched = True
                break

        if not matched:
            # If no match is found, add the character as a single-character token
            tokens.append(input_string[i])
            i += 1

    return tokens


def string_from_onehots(onehots):
    if len(onehots.shape) != 2:
        raise ValueError(f"onehots is not 2-D, but {len(onehots.shape)}-D {onehots.shape}")

    result = ""
    for onehot in onehots:
        result += VOCAB[np.argmax(onehot)]

    return result


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


def train_autoencoder(model, dataset, num_epochs=10, batch_size=16, val_split=0.2):
    """
    :type model: nn.Module

    :param model:
    :param dataset:
    :param num_epochs:
    :param val_split:
    :param batch_size:
    :return:
    """
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
                          lr=0.001,
                          momentum=0.9)

    for epoch in range(1, num_epochs+1):
        # Training
        train_loss = 0.
        model.train()

        train_start = time.time()
        train_progress = 0
        train_progress_needed = None

        print(f"Epoch {epoch}: Training...", end="")

        for sequence in train_loader:
            sequence = sequence.to(device)

            optimizer.zero_grad()

            outputs = model(sequence)
            W = sequence.shape[0]
            for w in range(W):
                if train_progress_needed is None:
                    train_progress_needed = len(train_loader) * W

                is_last = w == W - 1

                loss = criterion(outputs[w], sequence[w])
                loss.backward(retain_graph=not is_last)

                train_loss += loss.item()

                train_progress += 1

                print(f"\rEpoch {epoch}: Training... {train_progress/train_progress_needed:.1%} ETA {misc.timeformat(misc.timeleft(train_start, time.time(), train_progress, train_progress_needed))}", end="", flush=True)

            optimizer.step()

        # Validation
        val_loss = 0.
        model.eval()

        val_start = time.time()
        val_progress = 0
        val_progress_needed = None

        print(f"\rEpoch {epoch}: Validating...", end="", flush=True)

        for sequence in val_loader:
            sequence = sequence.to(device)

            outputs = model(sequence)
            for w in range(sequence.shape[0]):
                if val_progress_needed is None:
                    val_progress_needed = len(train_loader) * sequence.shape[0]

                loss = criterion(outputs[w], sequence[w])

                val_loss += loss.item()

                val_progress += 1

                print(
                    f"\rEpoch {epoch}: Validating... {val_progress / val_progress_needed:.1%} ETA {misc.timeformat(misc.timeleft(val_start, time.time(), val_progress, val_progress_needed))}",
                    end="", flush=True)

        print(
            f"\rEpoch: {epoch} Train Loss: {train_loss/len(train_loader):.4f} Val Loss: {val_loss/len(val_loader):.4f}"
        )
