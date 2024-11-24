import sys
import math

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

    # Compute gamma and digamma
    def compute_gammas(self, normalized_alphas, betas):
        num_states = len(self.transition_matrix)
        num_observations = len(self.observation_sequence)
        gammas = []
        digammas = []

        for time in range(num_observations - 1):
            gamma_t = [0] * num_states
            digamma_t = [[0] * num_states for _ in range(num_states)]
            denom = sum(normalized_alphas[time][i] * self.transition_matrix[i][j] *
                        self.emission_matrix[j][self.observation_sequence[time+1]] *
                        betas[time+1][j] for i in range(num_states) for j in range(num_states))
            denom = denom if denom != 0 else 1e-10

            for i in range(num_states):
                gamma_t[i] = 0
                for j in range(num_states):
                    digamma_t[i][j] = (normalized_alphas[time][i] * self.transition_matrix[i][j] *
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

        self.observation_sequence = observations
        if not self.observation_sequence:
            raise ValueError("No observations provided for training.")

        

        max_iterations = 100
        iterations = 0
        previous_log_prob = float('-inf')

        while iterations < max_iterations:
            # Forward and backward passes
            alphas, normalized_alphas, scaling_factors = self.forward_pass()
            betas = self.backward_pass(scaling_factors)

            # Compute gammas and digammas
            gammas, digammas = self.compute_gammas(normalized_alphas, betas)

            # Re-estimate parameters
            self.reestimate_parameters(gammas, digammas)

            # Compute log probability
            log_prob = -sum(math.log(c) for c in scaling_factors)
            if log_prob <= previous_log_prob:
                break
            previous_log_prob = log_prob
            iterations += 1

        # Output final matrices
        print(self.format_matrix(self.transition_matrix))
        print(self.format_matrix(self.emission_matrix))


    def format_matrix(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        flat = [round(val, 6) for row in matrix for val in row]
        return f"{rows} {cols} " + " ".join(map(str, flat))

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
    print("observations", observations)
    return observations

def question_7(observations):
    """
    Train an HMM model on provided observations.
    """
    # Initialize parameters
    A_initial = [[0.54, 0.26, 0.20],
                 [0.19, 0.53, 0.28],
                 [0.22, 0.18, 0.60]]
    
    B_initial = [[0.5, 0.2, 0.11, 0.19],
            [0.22, 0.28, 0.23, 0.27],
            [0.19, 0.21, 0.15, 0.45]]

    
    pi_initial = [0.3, 0.2, 0.5]

    # Initialize HMM
    hmm = HiddenMarkovModel(A_initial, B_initial, pi_initial)
    
    # Train the HMM
    hmm.train_model(observations)  # Assuming train_model accepts observations as an argument

    # Output trained parameters
    print("Trained parameters:")
    print("A:", hmm.transition_matrix)
    print("B:", hmm.emission_matrix)
    print("pi:", hmm.initial_distribution)

def main():
    """
    Main function to load data from stdin and run the HMM training.
    """
    observations = load_data_from_stdin()
    question_7(observations)

if __name__ == "__main__":
    main()
