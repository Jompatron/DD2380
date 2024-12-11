from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import numpy as np
import random
from collections import defaultdict


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        Initialize the Hidden Markov Model (HMM) parameters and data structures.
        
        Steps:
        1. Define the number of species, fish, and possible emissions.
        2. Initialize HMM probability matrices:
            - transition_probs: Probability of transitioning between states (species).
            - emission_probs: Probability of observing a particular emission given a state (species).
            - initial_probs: Probability distribution over initial states.
        3. Create data structures for collecting observations of each fish over time.
        4. Maintain count matrices for emissions and transitions that will be updated upon
           revelation of a fish's true species. These counts will help refine the HMM parameters.
        5. Set thresholds for the minimum number of observations needed before making a guess
           and a confidence threshold for making a guess.

        Notes:
        - Dirichlet distributions are used to initialize probabilities to ensure
          that probabilities sum to 1 and to break symmetry between states.
        - Laplace smoothing (adding a small constant, e.g., 0.1) is used in counts to avoid
          zero probabilities.
        - The confidence threshold and minimum number of observations are critical tuning parameters.
        """

        # Number of distinct species (hidden states in the HMM)
        self.n_species = N_SPECIES  
        # Number of fish (each fish sequence is modeled by the HMM)
        self.n_fish = N_FISH  
        # Number of possible emissions (observations)
        self.n_emissions = N_EMISSIONS

        # Initialize transition probabilities: Each row corresponds to a species,
        # each element in the row is the probability of transitioning to another species.
        # Using a Dirichlet distribution for random initialization to ensure a valid probability distribution.
        self.transition_probs = np.random.dirichlet(np.ones(self.n_species), size=self.n_species)

        # Initialize emission probabilities: Each row corresponds to a species,
        # and each element is the probability of an emission given that species.
        self.emission_probs = np.random.dirichlet(np.ones(self.n_emissions), size=self.n_species)

        # Initial state probabilities: Probability distribution over species at the first observation.
        self.initial_probs = np.random.dirichlet(np.ones(self.n_species))

        # Dictionary to store observations for each fish:
        # Key: fish_id, Value: list of observed emissions.
        self.observations = {fish_id: [] for fish_id in range(self.n_fish)}

        # Dictionary to store the revealed type of each fish once guessed:
        # Key: fish_id, Value: true species (once revealed).
        self.fish_types = {}

        # Matrices for counting occurrences of emissions and transitions:
        # Used to update probabilities after some are revealed.
        # Start with a small Laplace smoothing (0.1) to avoid zero probabilities.
        self.emission_counts = np.ones((self.n_species, self.n_emissions)) * 0.1
        self.transition_counts = np.ones((self.n_species, self.n_species)) * 0.1

        # Minimum number of observations required before making a guess for a particular fish.
        # For Kattis evaluation, this might be set to 80.
        self.min_observations = 80

        # Confidence threshold: Only guess if the most likely species probability
        # exceeds this threshold.
        self.confidence_threshold = 0.6

    def guess(self, step, observations):
        """
        Handle the guessing logic at each iteration.

        Parameters:
            step (int): The current iteration step.
            observations (list): A list of observed emissions for each fish at the current step.
                                 The list has length N_FISH, and each element is an integer
                                 representing the observed emission.

        Workflow:
        1. Update the observation sequences for each fish with the new emission observed at this step.
        2. Check if we have enough data for any fish that hasn't been guessed yet.
        3. If enough observations are collected, use the forward algorithm to compute posterior
           probabilities of each species for that fish.
        4. Identify the most likely species and check if the confidence exceeds the threshold.
        5. If confident enough, return (fish_id, species_guess). Otherwise, return None.

        Returns:
            None or (fish_id, fish_type): If we decide to guess, return the id of the fish and the
                                         guessed species. Otherwise, return None.
        """

        # Append the current observations to each fish's observation list
        for fish_id, obs in enumerate(observations):
            # Only track observations for fish that haven't been revealed/guessed yet
            if fish_id not in self.fish_types:
                self.observations[fish_id].append(obs)

        # Iterate over all fish to see if we can guess their type
        for fish_id, obs_seq in self.observations.items():
            # If this fish has not been revealed/guessed yet and we have enough observations
            if fish_id not in self.fish_types and len(obs_seq) >= self.min_observations:
                # Compute posterior probabilities of species using the forward algorithm
                species_probs = self._forward(obs_seq)
                # Find the species with the highest probability
                most_likely_species = np.argmax(species_probs)
                # Compute confidence as that species probability over the sum (should be normalized already)
                confidence = species_probs[most_likely_species] / np.sum(species_probs)

                # If our confidence exceeds the threshold, make a guess
                if confidence > self.confidence_threshold:
                    return fish_id, most_likely_species

        # If no guess is made, return None
        return None

    def reveal(self, correct, fish_id, true_type):
        """
        Called after making a guess. The game environment reveals whether we were correct,
        and provides the true species of the fish.

        Parameters:
            correct (bool): Indicates if the guess was correct.
            fish_id (int): The id of the fish for which we made a guess.
            true_type (int): The actual species of the fish.

        Steps:
        1. Record the revealed species in fish_types.
        2. Retrieve the observation sequence for that fish.
        3. Update the emission counts for that species using all the observations of this fish.
        4. Update the transition counts for that species based on the fish's observations.
        5. Update the HMM parameters (emission_probs, transition_probs, initial_probs) 
           using the updated counts.

        This feedback loop helps refine the model over time, making future guesses more accurate.
        """

        # Store the true species of the fish
        self.fish_types[fish_id] = true_type
        # Retrieve the observed sequence for that fish
        obs_seq = self.observations[fish_id]

        # Update emission counts: increment count of (true_type, observed_emission)
        for obs in obs_seq:
            self.emission_counts[true_type, obs] += 1

        # Update transition counts: since we are assuming a simplified model where transitions
        # occur within the same species over the sequence (a simplification),
        # we increment the self-transition counts for the species.
        for _ in range(len(obs_seq) - 1):
            self.transition_counts[true_type, true_type] += 1

        # Update probability matrices for this species based on the new counts
        self._update_probabilities(true_type)

    def _update_probabilities(self, species):
        """
        Update the probabilities for a given species based on the current counts.

        Steps:
        1. Update the emission probability distribution for the given species using emission_counts.
        2. Update the transition probability distribution for the given species using transition_counts.
        3. Update initial probabilities based on the distribution of known fish types.

        Parameters:
            species (int): The species index whose distributions we want to update.
        """

        # Update emission probabilities for the given species
        # Normalize the counts to get probabilities.
        self.emission_probs[species] = (
            self.emission_counts[species] / np.sum(self.emission_counts[species])
        )

        # Update transition probabilities for the given species
        self.transition_probs[species] = (
            self.transition_counts[species] / np.sum(self.transition_counts[species])
        )

        # Update initial probabilities based on the frequencies of observed species:
        # initial_probs is updated to reflect the distribution of species we have encountered so far.
        total_fish = len(self.fish_types)
        if total_fish > 0:
            for s in range(self.n_species):
                # Count how many fish of species s we have discovered
                species_count = sum(1 for t in self.fish_types.values() if t == s)
                self.initial_probs[s] = species_count / total_fish

    def _forward(self, obs_seq):
        """
        Implement the forward algorithm to compute the probability distribution over species
        at the last time step given an observation sequence.

        Parameters:
            obs_seq (list): A sequence of observed emissions for a single fish.

        Steps:
        1. Initialize alpha for t=0 using initial_probs and emission_probs.
        2. Recursively compute alpha for t > 0 using the formula:
           alpha[t, s] = emission_probs[s, obs_seq[t]] * sum_over_s'( alpha[t-1, s'] * transition_probs[s', s] )
        3. The final probabilities over species are given by alpha at the last time step.

        Returns:
            np.array: The vector of species probabilities at the final time step (size: n_species).
        """

        T = len(obs_seq)  # Length of the observation sequence
        alpha = np.zeros((T, self.n_species))

        # Initialization step: at time t=0
        for s in range(self.n_species):
            alpha[0, s] = self.initial_probs[s] * self.emission_probs[s, obs_seq[0]]

        # Forward pass: for each subsequent time step
        for t in range(1, T):
            for s in range(self.n_species):
                # Compute alpha[t, s] by summing over all possible previous states
                alpha[t, s] = self.emission_probs[s, obs_seq[t]] * sum(
                    alpha[t-1, s_prev] * self.transition_probs[s_prev, s]
                    for s_prev in range(self.n_species)
                )

        # Return the probability distribution over species at the last time step
        return alpha[-1]
