import sys
import math

class HMM:
    def __init__(self):
        self.A = []        # Transition matrix
        self.B = []        # Emission matrix
        self.Pi = []       # Initial state distribution
        self.O = []        # Observation sequence

    # Reads input from stdin and initializes A, B, Pi, and O
    def read_input(self):
        # Reads input from console, splits by line, and converts values to floats or ints
        A_in = self.string_change(sys.stdin.readline().split(), "float")
        B_in = self.string_change(sys.stdin.readline().split(), "float")
        Pi_in = self.string_change(sys.stdin.readline().split(), "float")
        O_in = self.string_change(sys.stdin.readline().split(), "int")
        self.O = O_in[1:]  # First value is the number of observations

        # Creates matrices A, B, and Pi
        self.A = self.create_matrix(A_in[2:], int(A_in[0]), int(A_in[1]))
        self.B = self.create_matrix(B_in[2:], int(B_in[0]), int(B_in[1]))
        self.Pi = self.create_matrix(Pi_in[2:], int(Pi_in[0]), int(Pi_in[1]))

    # Creates a matrix of size rows x columns
    def create_matrix(self, data, rows, columns):
        return [data[i:i+columns] for i in range(0, len(data), columns)]

    # Converts list of strings into floats or ints
    def string_change(self, A, choice):
        for i in range(len(A)):
            if choice == "float":
                A[i] = float(A[i])
            elif choice == "int":
                A[i] = int(A[i])
        return A

    # Forward algorithm (alpha-pass)
    def forward_algorithm(self):
        num_states = len(self.A)
        num_obvs = len(self.O)
        alphas = [[]]      # Holds the alphas for each observation
        alphas_normed = [[]]  # Holds the normalized alphas for each observation
        c = []

        # Computing alpha0
        c0 = 0
        Pi = self.Pi[0]  # Access the single row contained in Pi
        for i in range(num_states):  # Go through all states
            alpha0 = Pi[i] * self.B[i][self.O[0]]  # Using first observation
            alphas[0].append(alpha0)
            c0 += alpha0
        # Scaling
        c0 = 1 / c0 if c0 != 0 else 1e-10
        c.append(c0)
        alphas_normed[0] = [alphas[0][i]*c0 for i in range(len(alphas[0]))]

        # Computing alpha_t
        for t in range(1, num_obvs):  # Go through all observations except the first one
            c_t = 0
            current_alpha = []  # Alpha for this observation
            for i in range(num_states):
                alpha_t = 0
                for j in range(num_states):
                    alpha_t += alphas_normed[t-1][j] * self.A[j][i]
                alpha_t *= self.B[i][self.O[t]]
                current_alpha.append(alpha_t)
                c_t += alpha_t
            # Scaling
            c_t = 1 / c_t if c_t != 0 else 1e-10
            c.append(c_t)
            alphas.append(current_alpha)
            alphas_normed.append([current_alpha[i]*c_t for i in range(len(current_alpha))])

        return alphas, alphas_normed, c

    # Backward algorithm (beta-pass)
    def backward_algorithm(self, c):
        num_states = len(self.A)
        num_obvs = len(self.O)
        betas = [[] for _ in range(num_obvs)]

        # Initialize beta_T-1
        betas[num_obvs - 1] = [c[num_obvs - 1] for _ in range(num_states)]

        # Compute beta_t
        for t in range(num_obvs - 2, -1, -1):
            betas[t] = []
            for i in range(num_states):
                beta_t = 0
                for j in range(num_states):
                    beta_t += self.A[i][j] * self.B[j][self.O[t+1]] * betas[t+1][j]
                beta_t *= c[t]
                betas[t].append(beta_t)

        return betas

    # Compute gamma and digamma
    def compute_gamma(self, alphas_normed, betas):
        num_states = len(self.A)
        num_obvs = len(self.O)
        gammas = []
        digammas = []

        for t in range(num_obvs - 1):
            gamma_t = [0.0] * num_states
            digamma_t = [[0.0]*num_states for _ in range(num_states)]
            denom = 0.0
            for i in range(num_states):
                for j in range(num_states):
                    denom += alphas_normed[t][i] * self.A[i][j] * self.B[j][self.O[t+1]] * betas[t+1][j]
            denom = denom if denom != 0 else 1e-10
            for i in range(num_states):
                gamma_t[i] = 0.0
                for j in range(num_states):
                    digamma_t[i][j] = (alphas_normed[t][i] * self.A[i][j] * self.B[j][self.O[t+1]] * betas[t+1][j]) / denom
                    gamma_t[i] += digamma_t[i][j]
            gammas.append(gamma_t)
            digammas.append(digamma_t)

        # Special case for gamma_T-1
        gamma_T = alphas_normed[-1]
        gammas.append(gamma_T)

        return gammas, digammas

    # Re-estimate the model parameters A, B, and Pi
    def re_estimate(self, gammas, digammas):
        num_states = len(self.A)
        num_obvs = len(self.O)
        num_symbols = len(self.B[0])

        # Re-estimate Pi
        self.Pi = [gammas[0]]

        # Re-estimate A
        for i in range(num_states):
            denom = sum(gammas[t][i] for t in range(num_obvs - 1))
            denom = denom if denom != 0 else 1e-10
            for j in range(num_states):
                numer = sum(digammas[t][i][j] for t in range(num_obvs - 1))
                self.A[i][j] = numer / denom

        # Re-estimate B
        for i in range(num_states):
            denom = sum(gammas[t][i] for t in range(num_obvs))
            denom = denom if denom != 0 else 1e-10
            for k in range(num_symbols):
                numer = sum(gammas[t][i] for t in range(num_obvs) if self.O[t] == k)
                self.B[i][k] = numer / denom

    # Compute log probability
    def compute_log_prob(self, c):
        num_obvs = len(self.O)
        log_prob = -sum(math.log(ci) for ci in c)
        return log_prob

    # Format matrix for output
    def format_matrix(self, matrix):
        # Rounding elements
        for i in range(len(matrix)):
            matrix[i] = [round(elem, 6) for elem in matrix[i]]

        # Getting dimensions + formatting
        dimensions = f"{len(matrix)} {len(matrix[0])}"

        # Flattening
        flat_matrix = sum(matrix, [])

        # Formatting
        matrix_str = ' '.join(map(str, flat_matrix))

        # Combining dimensions + elements
        formatted_matrix = f"{dimensions} {matrix_str}"

        return formatted_matrix

    # Baum-Welch algorithm
    def baum_welch_algorithm(self):
        max_iters = 100
        iters = 0
        old_log_prob = -math.inf

        while True:
            # Forward pass
            alphas, alphas_normed, c = self.forward_algorithm()
            # Backward pass
            betas = self.backward_algorithm(c)
            # Compute gammas and digammas
            gammas, digammas = self.compute_gamma(alphas_normed, betas)
            # Re-estimate A, B, and Pi
            self.re_estimate(gammas, digammas)
            # Compute log probability
            log_prob = self.compute_log_prob(c)
            # Check for convergence
            iters += 1
            if iters >= max_iters or log_prob <= old_log_prob:
                break
            old_log_prob = log_prob

        # Prepare matrices for output
        A_formatted = self.format_matrix(self.A)
        B_formatted = self.format_matrix(self.B)

        # Output the matrices
        print(A_formatted)
        print(B_formatted)

def main():
    hmm = HMM()
    hmm.read_input()
    hmm.baum_welch_algorithm()

if __name__ == "__main__":
    main()
