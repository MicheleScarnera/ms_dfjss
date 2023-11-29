if __name__ == '__main__':
    import dfjss_nn
    dfjss_nn.generate_reward_model_file(batch_size=8,
                                        additional_constants_per_batch=4,
                                        num_batches=16000,
                                        seeds_per_batch=16,
                                        number_of_possible_seeds=16,
                                        features_weight_in_full_trees=5,
                                        max_depth=4)
