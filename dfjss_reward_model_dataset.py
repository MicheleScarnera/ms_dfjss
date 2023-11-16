if __name__ == '__main__':
    import dfjss_nn
    dfjss_nn.generate_reward_model_file(batch_size=8,
                                        num_batches=4000,
                                        seeds_per_batch=32,
                                        number_of_possible_seeds=128,
                                        features_weight_in_full_trees=5,
                                        max_depth=4)
