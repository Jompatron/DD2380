from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import numpy as np
import random
from collections import defaultdict


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        Initialize the HMM parameters with improved initial estimates and data structures
        for collecting statistics.
        """
        self.n_species = N_SPECIES
        self.n_fish = N_FISH
        self.n_emissions = N_EMISSIONS
        
        # Initialize with slight randomness to break symmetry
        self.transition_probs = np.random.dirichlet(np.ones(self.n_species), size=self.n_species)
        self.emission_probs = np.random.dirichlet(np.ones(self.n_emissions), size=self.n_species)
        self.initial_probs = np.random.dirichlet(np.ones(self.n_species))
        
        # Store observations and maintain counts for better probability estimates
        self.observations = {fish_id: [] for fish_id in range(self.n_fish)}
        self.fish_types = {}
        
        # Count matrices for updating probabilities
        self.emission_counts = np.ones((self.n_species, self.n_emissions)) * 0.1  # Laplace smoothing
        self.transition_counts = np.ones((self.n_species, self.n_species)) * 0.1
        #self.species_counts = defaultdict(lambda: defaultdict(int))
        
        # Minimum observations before making a guess
        # For Kattis, set this to 80
        self.min_observations = 10
        
        # Confidence threshold for making guesses
        self.confidence_threshold = 0.6

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        # Update observation sequences
        # Use fish_id as index and append new observations to this fish's sequence
        for fish_id, obs in enumerate(observations):
            if fish_id not in self.fish_types:
                self.observations[fish_id].append(obs)

        # Only guess if we have enough observations
        for fish_id, obs_seq in self.observations.items():
            if fish_id not in self.fish_types and len(obs_seq) >= self.min_observations:
                # Use forward algorithm to compute species probabilities
                species_probs = self._forward(obs_seq)
                most_likely_species = np.argmax(species_probs)
                confidence = species_probs[most_likely_species] / np.sum(species_probs)
                
                # Only guess if confidence exceeds threshold
                if confidence > self.confidence_threshold:
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
        """
        self.fish_types[fish_id] = true_type
        #print(f"Revealed fish {fish_id} as type {true_type}")
        obs_seq = self.observations[fish_id]
        
        # Update emission counts
        for obs in obs_seq:
            self.emission_counts[true_type, obs] += 1
        
        # Update transition counts
        for _ in range(len(obs_seq) - 1):
            #curr_obs = obs_seq[i]
            #next_obs = obs_seq[i + 1]
            self.transition_counts[true_type, true_type] += 1
        
        # Update probability matrices
        self._update_probabilities(true_type)
    
    def _update_probabilities(self, species):
        """
        Update probability matrices based on accumulated counts.
        """
        # Update emission probabilities
        self.emission_probs[species] = (
            self.emission_counts[species] / np.sum(self.emission_counts[species])
        )
        
        # Update transition probabilities
        self.transition_probs[species] = (
            self.transition_counts[species] / np.sum(self.transition_counts[species])
        )
        
        # Update initial probabilities based on observed species distribution
        total_fish = len(self.fish_types)
        if total_fish > 0:
            for s in range(self.n_species):
                self.initial_probs[s] = (
                    sum(1 for t in self.fish_types.values() if t == s) / total_fish
                )

    def _forward(self, obs_seq):
        """
        Implement forward algorithm for computing species probabilities.
        """
        T = len(obs_seq)
        alpha = np.zeros((T, self.n_species))
        
        # Initialize first time step
        for s in range(self.n_species):
            alpha[0, s] = self.initial_probs[s] * self.emission_probs[s, obs_seq[0]]
        
        # Forward pass
        for t in range(1, T):
            for s in range(self.n_species):
                alpha[t, s] = self.emission_probs[s, obs_seq[t]] * sum(
                    alpha[t-1, s2] * self.transition_probs[s2, s]
                    for s2 in range(self.n_species)
                )
        
        return alpha[-1] # Return probabilities for last time step
