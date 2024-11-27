from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import numpy as np
import random


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        Initialize HMM parameters.
        """
        # Number of species and fish
        self.n_species = N_SPECIES
        self.n_fish = N_FISH

        # Initialize transition, emission, and initial probabilities
        self.transition_probs = np.full((self.n_species, self.n_species), 1 / self.n_species)
        self.emission_probs = np.full((self.n_species, N_STEPS), 1 / N_STEPS)
        self.initial_probs = np.full(self.n_species, 1 / self.n_species)

        # Store observations for each fish
        self.observations = {fish_id: [] for fish_id in range(self.n_fish)}

        # Known fish types
        self.fish_types = {}

    def guess(self, step, observations):
        """
        Process observations and make a guess.
        """
        for fish_id, obs in enumerate(observations):
            self.observations[fish_id].append(obs)

        # Example: Use the most common observation to guess fish type
        for fish_id, obs_seq in self.observations.items():
            if fish_id not in self.fish_types and len(obs_seq) > 5:  # Require a few observations to guess
                most_likely_species = self._viterbi(obs_seq)
                return fish_id, most_likely_species

        return None

    def reveal(self, correct, fish_id, true_type):
        """
        Update model with revealed fish type.
        """
        self.fish_types[fish_id] = true_type

        # Update emission probabilities
        for obs in self.observations[fish_id]:
            self.emission_probs[true_type, obs] += 1

        # Normalize probabilities
        self.emission_probs[true_type] /= np.sum(self.emission_probs[true_type])

    def _viterbi(self, obs_seq):
        """
        Viterbi algorithm to find the most likely species.
        """
        T = len(obs_seq)
        dp = np.zeros((self.n_species, T))
        backtrack = np.zeros((self.n_species, T), dtype=int)

        # Initialize base cases
        for s in range(self.n_species):
            dp[s, 0] = self.initial_probs[s] * self.emission_probs[s, obs_seq[0]]

        # Fill DP table
        for t in range(1, T):
            for s in range(self.n_species):
                probabilities = [
                    dp[prev_s, t - 1] * self.transition_probs[prev_s, s] * self.emission_probs[s, obs_seq[t]]
                    for prev_s in range(self.n_species)
                ]
                dp[s, t] = max(probabilities)
                backtrack[s, t] = np.argmax(probabilities)

        # Backtrack to find the most likely sequence
        most_likely_species = np.argmax(dp[:, -1])
        return most_likely_species
