import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np
import pandas as pd

import time
import os

import dfjss_genetic as genetic
import dfjss_misc as misc

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
VOCAB = ["(", ")", "+", "-", "*", "/", "<", ">", *INDIVIDUALS_FEATURES]


class IndividualAutoEncoder(nn.Module):
    def __init__(self, input_size=None, hidden_size=128, num_layers=1, dropout=0, bidirectional=False):
        super(IndividualAutoEncoder, self).__init__()

        if input_size is None:
            input_size = len(VOCAB)

        self.encoder = nn.RNN(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              device=device)

        d = 2 if bidirectional else 1
        self.encoder_output_size = d * num_layers * hidden_size

        self.decoder = nn.RNN(input_size=self.encoder_output_size,
                              hidden_size=input_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=False,
                              device=device)

    def forward(self, x):
        vocab_length = self.encoder.input_size
        sequence_length = x.size()[0]

        _, encoded = self.encoder(x)

        current_hidden_state = torch.zeros((1, vocab_length))

        decodes = []
        for i in range(sequence_length):
            d, current_hidden_state = self.decoder(encoded, current_hidden_state)

            decodes.append(d)

        return torch.stack(decodes)


def generate_individuals_file(total_amount=500000, max_depth=8, rng_seed=100):
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


class IndividualDataset(data.IterableDataset):
    def __init__(self):
        super().__init__()

    def __iter__(self):
        return self.data_iterator()

    def data_iterator(self):
        # Load the CSV file using pandas
        if os.path.exists(INDIVIDUALS_FILENAME):
            df = pd.read_csv(INDIVIDUALS_FILENAME)
        else:
            df = generate_individuals_file()

        # Iterate through each row in the CSV
        for _, row in df.iterrows():
            individual_sequence = row['Individual']

            # Convert the string to a list of one-hot vectors
            one_hot_sequence = [self.string_to_one_hot(s) for s in individual_sequence]

            yield torch.stack(one_hot_sequence)

    def string_to_one_hot(self, s):
        # Convert a single character to a one-hot tensor
        one_hot = torch.zeros(len(VOCAB))
        if s in VOCAB:
            index = VOCAB.index(s)
            one_hot[index] = 1
        return one_hot
