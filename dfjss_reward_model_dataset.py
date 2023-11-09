if __name__ == '__main__':
    import dfjss_nn
    dfjss_nn.generate_reward_model_file(batch_size=128, simulation_seeds_amount=8, features_weight_in_full_trees=5, max_depth=4)
