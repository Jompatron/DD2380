#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import numpy as np
import math




class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):

        #self.transition_matrix = []   # Transition probabilities
        #self.emission_matrix = []    # Emission probabilities
        #self.initial_distribution = []  # Initial state probabilities
        #self.observation_sequence = []  # Sequence of observations
        """
        Initialize HMM parameters.
        """
        # Number of species and fish
        self.n_species = N_SPECIES
        self.n_fish = N_FISH

        # Initialize transition, emission, and initial probabilities
        self.transition_matrix = np.full((self.n_species, self.n_species), 1 / self.n_species)
        self.emission_matrix = np.full((self.n_species, N_STEPS), 1 / N_STEPS)
        self.initial_distribution = np.full(self.n_species, 1 / self.n_species)

        # Store observations for each fish
        self.observations = {fish_id: [] for fish_id in range(self.n_fish)}

        # Known fish types
        self.fish_types = {}


        self.observation_sequence = []
        self.observation_sequences = []

        for fish_id in self.fish_types:
            self.observation_sequence.extend(self.observations[fish_id])

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        # This code would make a random guess on each step:
        # return (step % N_FISH, random.randint(0, N_SPECIES - 1))

        """
        Process observations and make a guess.
        """
        for fish_id, obs in enumerate(observations):
            self.observations[fish_id].append(obs)  # Collect observations for each fish

        # Use Viterbi for guessing fish types
        for fish_id, obs_seq in self.observations.items():
            if fish_id not in self.fish_types and len(obs_seq) > 5:
                species_probabilities = self.compute_species_probabilities(obs_seq)
                most_likely_species = np.argmax(species_probabilities)
                confidence = species_probabilities[most_likely_species]
                if confidence > 1.5:
                    return fish_id, most_likely_species
        return None

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        
        Update model with revealed fish type.
        """
        self.fish_types[fish_id] = true_type
        self.observation_sequence.extend(self.observations[fish_id])
        self.observation_sequences.append(self.observations[fish_id])
        self.train_model()

    def compute_species_probabilities(self, obs_seq):
        num_states = self.n_species
        num_steps = len(obs_seq)
        log_transition = np.log(self.transition_matrix + 1e-10)
        log_emission = np.log(self.emission_matrix + 1e-10)
        log_initial = np.log(self.initial_distribution + 1e-10)
        dp = np.zeros((num_states, num_steps))

        # Initialization
        dp[:, 0] = log_initial + log_emission[:, obs_seq[0]]

        # Recursion
        for t in range(1, num_steps):
            for s in range(num_states):
                dp[s, t] = np.max(dp[:, t - 1] + log_transition[:, s]) + log_emission[s, obs_seq[t]]

        # Compute probabilities
        final_probs = np.exp(dp[:, -1])
        total_prob = np.sum(final_probs)
        species_probabilities = final_probs / total_prob

        return species_probabilities

    def update_parameters_with_labeled_data(self):
        num_states = self.n_species
        num_symbols = self.emission_matrix.shape[1]

        # Initialize counts
        emission_counts = np.zeros((num_states, num_symbols))
        transition_counts = np.zeros((num_states, num_states))

        # Count emissions and transitions for each species
        for species_id, sequences in self.species_observations.items():
            for seq in sequences:
                for t in range(len(seq)):
                    emission_counts[species_id, seq[t]] += 1
                    if t > 0:
                        transition_counts[species_id, species_id] += 1  # Assuming self-transition

        # Normalize to get probabilities
        self.emission_matrix = (emission_counts.T / emission_counts.sum(axis=1)).T
        self.transition_matrix = (transition_counts.T / transition_counts.sum(axis=1)).T



    def run_viterbi(self, obs_seq):
        num_states = len(self.transition_matrix)
        num_steps = len(obs_seq)

        log_transition = np.log(self.transition_matrix + 1e-10)
        log_emission = np.log(self.emission_matrix + 1e-10)
        log_initial = np.log(self.initial_distribution + 1e-10)

        dp = np.zeros((num_states, num_steps))
        backpointer = np.zeros((num_states, num_steps), dtype=int)

        # Initialization
        dp[:, 0] = log_initial + log_emission[:, obs_seq[0]]

        # Recursion
        for t in range(1, num_steps):
            for s in range(num_states):
                prob = dp[:, t - 1] + log_transition[:, s] + log_emission[s, obs_seq[t]]
                dp[s, t] = np.max(prob)
                backpointer[s, t] = np.argmax(prob)

        # Termination
        most_likely_final_state = np.argmax(dp[:, -1])

        return most_likely_final_state

    def forward_pass(self):
        num_states = len(self.transition_matrix)
        num_observations = len(self.observation_sequence)
        scaling_factors = np.zeros(num_observations)

        # Initialize alpha array
        alpha = np.zeros((num_observations, num_states))

        # Initialization step
        alpha[0, :] = self.initial_distribution * self.emission_matrix[:, self.observation_sequence[0]]
        scaling_factors[0] = np.sum(alpha[0, :])

        # Scale alpha[0, :]
        if scaling_factors[0] == 0:
            scaling_factors[0] = 1e-10  # Avoid division by zero
        alpha[0, :] /= scaling_factors[0]

        # Recursion step
        for t in range(1, num_observations):
            alpha[t, :] = (alpha[t - 1, :] @ self.transition_matrix) * self.emission_matrix[:, self.observation_sequence[t]]
            scaling_factors[t] = np.sum(alpha[t, :])

            # Scale alpha[t, :]
            if scaling_factors[t] == 0:
                scaling_factors[t] = 1e-10
            alpha[t, :] /= scaling_factors[t]

        return alpha, scaling_factors


    # Backward pass (beta computation)
    def backward_pass(self, scaling_factors):
        num_states = len(self.transition_matrix)
        num_observations = len(self.observation_sequence)
        beta = np.zeros((num_observations, num_states))

        # Initialization step
        beta[-1, :] = 1 / scaling_factors[-1]

        # Recursion step
        for t in range(num_observations - 2, -1, -1):
            beta[t, :] = (self.transition_matrix @ (self.emission_matrix[:, self.observation_sequence[t + 1]] * beta[t + 1, :])) / scaling_factors[t]

        return beta



    # Compute gamma and digamma
    def compute_gammas(self, alpha, beta):
        num_states = len(self.transition_matrix)
        num_observations = len(self.observation_sequence)
        gamma = np.zeros((num_observations, num_states))
        di_gamma = np.zeros((num_observations - 1, num_states, num_states))

        for t in range(num_observations - 1):
            denominator = np.sum(alpha[t, :] * beta[t, :])
            if denominator == 0:
                denominator = 1e-10

            for i in range(num_states):
                gamma[t, i] = (alpha[t, i] * beta[t, i]) / denominator
                for j in range(num_states):
                    di_gamma[t, i, j] = (alpha[t, i] * self.transition_matrix[i, j] *
                                        self.emission_matrix[j, self.observation_sequence[t + 1]] *
                                        beta[t + 1, j]) / denominator

        # Special case for gamma at time T - 1
        denominator = np.sum(alpha[-1, :] * beta[-1, :])
        if denominator == 0:
            denominator = 1e-10
        gamma[-1, :] = (alpha[-1, :] * beta[-1, :]) / denominator

        return gamma, di_gamma


    # Re-estimate model parameters
    def reestimate_parameters(self, gammas, digammas):
        num_states = len(self.transition_matrix)
        num_symbols = len(self.emission_matrix[0])

        # Re-estimate initial distribution
        self.initial_distribution = [gammas[0]]

        # Re-estimate transition matrix
        for i in range(num_states):
            denom = sum(gammas[t][i] for t in range(len(gammas) - 1))
            denom = denom if denom != 0 else 1e-10
            for j in range(num_states):
                numer = sum(digammas[t][i][j] for t in range(len(digammas)))
                self.transition_matrix[i][j] = numer / denom

        # Re-estimate emission matrix
        for i in range(num_states):
            denom = sum(gammas[t][i] for t in range(len(gammas)))
            denom = denom if denom != 0 else 1e-10
            for k in range(num_symbols):
                numer = sum(gammas[t][i] for t in range(len(gammas)) if self.observation_sequence[t] == k)
                self.emission_matrix[i][k] = numer / denom

    # Baum-Welch algorithm
    def train_model(self):
        max_iterations = 100
        iterations = 0
        previous_log_prob = float('-inf')

        while iterations < max_iterations:
            # Initialize accumulators for re-estimation
            accum_transition = np.zeros_like(self.transition_matrix)
            accum_emission = np.zeros_like(self.emission_matrix)
            accum_initial = np.zeros_like(self.initial_distribution)

            for seq in self.observation_sequences:
                self.observation_sequence = seq
                # Forward and backward passes
                alpha, scaling_factors = self.forward_pass()
                betas = self.backward_pass(scaling_factors)

                # Compute gammas and digammas
                gammas, digammas = self.compute_gammas(alpha, betas)

                # Re-estimate parameters
                self.reestimate_parameters(gammas, digammas)

                # Compute log probability
                log_prob = -sum(math.log(c) for c in scaling_factors)
                if log_prob <= previous_log_prob:
                    break
                previous_log_prob = log_prob
                iterations += 1

