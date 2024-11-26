import sys
import math
from copy import deepcopy

class Distribution:
    def __init__(self, start_from_goal_dist=False, no_hidden_states=3):
        self.start_from_goal_dist = start_from_goal_dist
        self.A_init = []
        self.B_init = []
        self.pi_init = []
        self.no_hidden_states = no_hidden_states
        self.set_distribution()

    def set_distribution(self):
        if self.no_hidden_states == 2:
            # Add a specific configuration for 2 hidden states
            self.A_init = [[0.7, 0.3],
                        [0.2, 0.8]]
            
            self.B_init = [[0.6, 0.2, 0.1, 0.1],
                        [0.1, 0.3, 0.3, 0.3]]
            
            self.pi_init = [0.5, 0.5]

        elif self.start_from_goal_dist and self.no_hidden_states == 3:
            # Existing 3-state configuration when start_from_goal_dist is True
            self.A_init = [[0.7, 0.05, 0.25],
                            [0.1, 0.8, 0.1],
                            [0.2, 0.3, 0.5]]
        
            self.B_init = [[0.7, 0.2, 0.1, 0.0],
                            [0.1, 0.4, 0.3, 0.2],
                            [0, 0.1, 0.2, 0.7]]
        
            self.pi_init = [0.1, 0, 0]

        elif not self.start_from_goal_dist and self.no_hidden_states == 3:
            # Existing 3-state configuration when start_from_goal_dist is False
            self.A_init = [[0.54, 0.26, 0.20],
                            [0.19, 0.53, 0.28],
                            [0.22, 0.18, 0.60]]
        
            self.B_init = [[0.5, 0.2, 0.11, 0.19],
                            [0.22, 0.28, 0.23, 0.27],
                            [0.19, 0.21, 0.15, 0.45]]
        
            self.pi_init = [0.3, 0.2, 0.5]

        else:  # More than 3 states or other configurations
            # Existing logic for 4 or 5 states
            self.A_init = [[0.7, 0.05, 0.15, 0.05, 0.05],
                            [0.1, 0.8, 0.05, 0.05, 0],
                            [0.1, 0.1, 0.7, 0.1, 0],
                            [0.05, 0.05, 0.05, 0.8, 0.05],
                            [0.05, 0, 0.1, 0.05, 0.8]]
        
            self.B_init = [[0.7, 0.2, 0.1, 0, 0],
                            [0.1, 0.4, 0.3, 0.2, 0],
                            [0, 0.1, 0.2, 0.7, 0],
                            [0, 0, 0, 0.8, 0.2],
                            [0, 0, 0, 0.2, 0.8]]
        
            self.pi_init = [0.1, 0.15, 0.15, 0.4, 0.2]

            # Sample number of hidden states rows and columns from A and B
            self.A_init = self.A_init[:self.no_hidden_states]
            self.B_init = self.B_init[:self.no_hidden_states]
            self.A_init = [row[:self.no_hidden_states] for row in self.A_init]
            self.B_init = [row[:self.no_hidden_states] for row in self.B_init]
            self.pi_init = self.pi_init[:self.no_hidden_states]


class HiddenMarkovModel:
    def __init__(self, A=None, B=None, pi=None, observations=None):
        self.transition_matrix = A if A else []  # Transition probabilities
        self.emission_matrix = B if B else []    # Emission probabilities
        self.initial_distribution = pi if pi else []  # Initial state probabilities
        self.observation_sequence = observations if observations else []  # Sequence of observations

    # Read input and initialize matrices
    def load_input(self):
        """
        Load observations from stdin for both single-row and multi-row input.
        """
        data = sys.stdin.read().strip().splitlines()
        observations = []

        for line in data:
            # Split the line into integers and skip the first value (the count)
            parsed_line = list(map(int, line.split()))
            observations.extend(parsed_line[1:])  # Skip the first value in each line

        self.observation_sequence = observations

    # Create a matrix of given dimensions
    def _create_matrix(self, values, rows, cols):
        return [values[i:i+cols] for i in range(0, len(values), cols)]

    # Parse input and convert to the desired type
    def _parse_input(self, raw_data, data_type):
        if data_type == "float":
            return [float(item) for item in raw_data]
        elif data_type == "int":
            return [int(item) for item in raw_data]

    # Forward pass (alpha computation)
    def forward_pass(self):
        num_states = len(self.transition_matrix)
        num_observations = len(self.observation_sequence)
        alpha_values = [[]]
        normalized_alpha = [[]]
        scaling_factors = []

        # Compute alpha_0
        initial_scale = 0
        for state in range(num_states):
            alpha_0 = self.initial_distribution[state] * self.emission_matrix[state][self.observation_sequence[0]]
            alpha_values[0].append(alpha_0)
            initial_scale += alpha_0

        # Normalize alpha_0
        scaling_factor_0 = 1 / initial_scale if initial_scale != 0 else 1e-10
        scaling_factors.append(scaling_factor_0)
        normalized_alpha[0] = [alpha * scaling_factor_0 for alpha in alpha_values[0]]

        # Compute alpha_t
        for time in range(1, num_observations):
            scale_t = 0
            current_alpha = []
            for state in range(num_states):
                alpha_t = sum(normalized_alpha[time-1][prev_state] * self.transition_matrix[prev_state][state]
                              for prev_state in range(num_states))
                alpha_t *= self.emission_matrix[state][self.observation_sequence[time]]
                current_alpha.append(alpha_t)
                scale_t += alpha_t
            # Normalize
            scaling_factor_t = 1 / scale_t if scale_t != 0 else 1e-10
            scaling_factors.append(scaling_factor_t)
            alpha_values.append(current_alpha)
            normalized_alpha.append([alpha * scaling_factor_t for alpha in current_alpha])

        return alpha_values, normalized_alpha, scaling_factors


    # Backward pass (beta computation)
    def backward_pass(self, scaling_factors):
        num_states = len(self.transition_matrix)
        num_observations = len(self.observation_sequence)
        beta_values = [[scaling_factors[-1]] * num_states for _ in range(num_observations)]

        # Compute beta_t
        for time in range(num_observations - 2, -1, -1):
            for state in range(num_states):
                beta_values[time][state] = sum(self.transition_matrix[state][next_state] *
                                               self.emission_matrix[next_state][self.observation_sequence[time+1]] *
                                               beta_values[time+1][next_state]
                                               for next_state in range(num_states))
                beta_values[time][state] *= scaling_factors[time]

        return beta_values

    def compute_gammas(self, normalized_alphas, betas):
        num_states = len(self.transition_matrix)
        num_observations = len(self.observation_sequence)
        gammas = []
        digammas = []

        for time in range(num_observations - 1):
            gamma_t = [0] * num_states
            digamma_t = [[0] * num_states for _ in range(num_states)]
            
            # Improved denominator calculation
            denom = 0
            for i in range(num_states):
                for j in range(num_states):
                    temp = (normalized_alphas[time][i] * 
                            self.transition_matrix[i][j] *
                            self.emission_matrix[j][self.observation_sequence[time+1]] *
                            betas[time+1][j])
                    denom += temp

            denom = max(denom, 1e-10)  # Prevent division by zero

            for i in range(num_states):
                for j in range(num_states):
                    digamma_t[i][j] = (normalized_alphas[time][i] * 
                                    self.transition_matrix[i][j] *
                                    self.emission_matrix[j][self.observation_sequence[time+1]] *
                                    betas[time+1][j]) / denom
                    gamma_t[i] += digamma_t[i][j]
            
            gammas.append(gamma_t)
            digammas.append(digamma_t)

        # Special case for the last gamma
        gammas.append(normalized_alphas[-1])

        return gammas, digammas

    # Re-estimate model parameters
    def reestimate_parameters(self, gammas, digammas):
        num_states = len(self.transition_matrix)
        num_symbols = len(self.emission_matrix[0])

        # Re-estimate initial distribution
        self.initial_distribution = gammas[0]  # Fix: Ensure 1D list

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

    def train_model(self, observations):
        max_iterations = 100000
        iterations = 0
        previous_log_prob = float('-inf')
        convergence_threshold = 1e-10

        log_probabilities = []
        transition_matrix_history = [deepcopy(self.transition_matrix)]
        emission_matrix_history = [deepcopy(self.emission_matrix)]
        initial_distribution_history = [deepcopy(self.initial_distribution)]

        while iterations < max_iterations:
            # Forward and backward passes
            alphas, normalized_alphas, scaling_factors = self.forward_pass()
            betas = self.backward_pass(scaling_factors)

            # Compute gammas and digammas
            gammas, digammas = self.compute_gammas(normalized_alphas, betas)

            # Store previous matrices before re-estimation
            prev_transition_matrix = deepcopy(self.transition_matrix)
            prev_emission_matrix = deepcopy(self.emission_matrix)
            prev_initial_distribution = deepcopy(self.initial_distribution)

            # Re-estimate parameters
            self.reestimate_parameters(gammas, digammas)

            # Compute log probability
            log_prob = -sum(math.log(c) for c in scaling_factors)
            log_probabilities.append(log_prob)

            # Detailed parameter change tracking
            transition_change = self._compute_matrix_change(prev_transition_matrix, self.transition_matrix)
            emission_change = self._compute_matrix_change(prev_emission_matrix, self.emission_matrix)
            initial_dist_change = self._compute_vector_change(prev_initial_distribution, self.initial_distribution)

            # Store matrices for history
            transition_matrix_history.append(deepcopy(self.transition_matrix))
            emission_matrix_history.append(deepcopy(self.emission_matrix))
            initial_distribution_history.append(deepcopy(self.initial_distribution))

            # Verbose output
            #print(f"Iteration {iterations}:")
            #print(f"Log Probability: {log_prob}")
            #print(f"Transition Matrix Change: {transition_change}")
            #print(f"Emission Matrix Change: {emission_change}")
            #print(f"Initial Distribution Change: {initial_dist_change}")
            #print("---")

            # Convergence check with more robust criteria
            if (iterations > 0 and 
                transition_change < convergence_threshold and 
                emission_change < convergence_threshold and 
                initial_dist_change < convergence_threshold):
                print("Converged based on parameter changes")
                break

            # Check log probability change
            if iterations > 0 and abs(log_prob - previous_log_prob) < convergence_threshold:
                print("Converged based on log probability")
                break

            previous_log_prob = log_prob
            iterations += 1

        # Print final matrices for comparison
        print("\nFinal Transition Matrix:")
        self.format_matrix(self.transition_matrix)
        print("\nFinal Emission Matrix:")
        self.format_matrix(self.emission_matrix)
        print("\nFinal Initial Distribution:")
        print(" ".join(map(str, self.initial_distribution)))

        print(f"\nConverged after {iterations} iterations.")
        return log_probabilities

    def _compute_matrix_change(self, old_matrix, new_matrix):
        """
        Compute the maximum absolute change between two matrices
        """
        if not old_matrix or not new_matrix:
            return float('inf')
        
        max_change = 0
        for i in range(len(old_matrix)):
            for j in range(len(old_matrix[0])):
                max_change = max(max_change, abs(old_matrix[i][j] - new_matrix[i][j]))
        
        return max_change

    def _compute_vector_change(self, old_vector, new_vector):
        """
        Compute the maximum absolute change between two vectors
        """
        if not old_vector or not new_vector:
            return float('inf')
        
        return max(abs(old - new) for old, new in zip(old_vector, new_vector))
    
    def format_matrix(self, matrix):
        """
        Prints out the matrix in a matrix format.
        """
        rows = len(matrix)
        cols = len(matrix[0])
        for i in range(rows):
            for j in range(cols):
                print(f"{matrix[i][j]:.6f}", end=" ")
            print()

def load_data_from_stdin():
    """
    Load HMM observation data from standard input.
    """
    observations = []
    for line in sys.stdin:
        # Each line contains space-separated observation indices
        observations.extend(map(int, line.strip().split()))
    valid_observation_symbols = [0, 1, 2, 3]
    observations = [o for o in observations if o in valid_observation_symbols]
    return observations

def try_different_numbers_of_hidden_states(observations):
    """
    Try different numbers of hidden states for the HMM.
    """
    for num_states in [2, 3, 4, 5]:  # You can adjust this range
        initial_dist = Distribution(start_from_goal_dist=True, no_hidden_states=num_states)
        hmm = HiddenMarkovModel(initial_dist.A_init, initial_dist.B_init, initial_dist.pi_init)
        hmm.load_input()
        
        print(f"\n--- Training with {num_states} Hidden States ---")
        hmm.train_model(observations)

def create_uniform_distribution(num_states, num_symbols):
    """
    Create uniform distributions for A, B, and π
    """
    # Uniform transition matrix (each row sums to 1)
    A_uniform = [[1/num_states for _ in range(num_states)] for _ in range(num_states)]
    
    # Uniform emission matrix (each row sums to 1)
    B_uniform = [[1/num_symbols for _ in range(num_symbols)] for _ in range(num_states)]
    
    # Uniform initial distribution
    pi_uniform = [1/num_states for _ in range(num_states)]
    
    return A_uniform, B_uniform, pi_uniform

def create_diagonal_distribution(num_states, num_symbols):
    """
    Create a diagonal transition matrix with π = [0, 0, 1] for a 3-state model
    """
    # Diagonal transition matrix
    A_diagonal = [[1 if i == j else 0 for j in range(num_states)] for i in range(num_states)]
    
    # Uniform emission matrix (each row sums to 1)
    B_diagonal = [[1/num_symbols for _ in range(num_symbols)] for _ in range(num_states)]
    
    # Specific initial distribution [0, 0, 1]
    pi_diagonal = [1 if i == num_states-1 else 0 for i in range(num_states)]
    
    return A_diagonal, B_diagonal, pi_diagonal

def create_goal_similar_distribution(num_states, num_symbols):
    """
    Create matrices close to the goal distribution
    """
    # Similar to the goal distribution, but slightly perturbed
    A_similar = [
        [0.6, 0.1, 0.3],
        [0.15, 0.7, 0.15],
        [0.25, 0.25, 0.5]
    ][:num_states]
    A_similar = [row[:num_states] for row in A_similar]
    
    B_similar = [
        [0.6, 0.2, 0.1, 0.1],
        [0.2, 0.3, 0.3, 0.2],
        [0.1, 0.1, 0.2, 0.6]
    ][:num_states]
    B_similar = [row[:num_symbols] for row in B_similar]
    
    # Initial distribution close to goal
    pi_similar = [0.2, 0.1, 0.7][:num_states]
    
    return A_similar, B_similar, pi_similar

def explore_initialization_strategies(observations, num_states=3):
    """
    Explore different initialization strategies for Baum-Welch algorithm
    """
    strategies = [
        ("Uniform Distribution", create_uniform_distribution),
        ("Diagonal Matrix with π[0,0,1]", create_diagonal_distribution),
        ("Close to Goal Distribution", create_goal_similar_distribution)
    ]
    
    for strategy_name, init_func in strategies:
        print(f"\n{'=' * 50}")
        print(f"Exploring: {strategy_name}")
        print(f"Number of Hidden States: {num_states}")
        print(f"{'=' * 50}")
        
        # Create initial distributions
        A_init, B_init, pi_init = init_func(num_states, 4)  # 4 observation symbols
        
        # Create and train HMM
        hmm = HiddenMarkovModel(A_init, B_init, pi_init)
        hmm.observation_sequence = observations
        
        print("\nInitial Transition Matrix (A):")
        hmm.format_matrix(hmm.transition_matrix)
        
        print("\nInitial Emission Matrix (B):")
        hmm.format_matrix(hmm.emission_matrix)
        
        print("\nInitial Distribution (π):")
        print(" ".join(map(str, hmm.initial_distribution)))
        
        # Train the model
        hmm.train_model(observations)


def main():
    """
    Main function to load data from stdin and run the HMM training.
    """
    observations = load_data_from_stdin()

    #try_different_numbers_of_hidden_states(observations)
    for num_states in [3]:
        explore_initialization_strategies(observations, num_states)

def explore_hidden_states(observations):
    """
    Explore the impact of different numbers of hidden states
    """
    goal_dist_options = [True, False]
    state_options = [2, 3, 4, 5]
    
    for goal_dist in goal_dist_options:
        print(f"\n{'=' * 20}")
        print(f"Exploring with start_from_goal_dist = {goal_dist}")
        print(f"{'=' * 20}")
        
        for num_states in state_options:
            initial_dist = Distribution(start_from_goal_dist=goal_dist, no_hidden_states=num_states)
            hmm = HiddenMarkovModel(initial_dist.A_init, initial_dist.B_init, initial_dist.pi_init)
            hmm.load_input()
            
            print(f"\n--- Training with {num_states} Hidden States ---")
            hmm.train_model(observations)

if __name__ == "__main__":
    main()
