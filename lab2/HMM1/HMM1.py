import sys

class HMM:
    def __init__(self):
        self.transition_matrix = []
        self.emission_matrix = []
        self.initial_state = []
        self.emissions = []
        self.alphas = []

    def read_input(self):
        data = sys.stdin.read().strip().split('\n')
        self.transition_matrix = self.read_matrix(data[0])
        self.emission_matrix = self.read_matrix(data[1])
        self.initial_state = self.read_vector(data[2])
        self.emissions = self.read_emissions(data[3])

    def read_emissions(self, line):
        parts = list(map(int, line.split()))
        return parts[1:]
    
    def read_matrix(self, line):
        parts = list(map(float, line.split()))
        rows, cols = int(parts[0]), int(parts[1])
        values = parts[2:]
        return [values[i * cols:(i + 1) * cols] for i in range(rows)]

    def read_vector(self, line):
        parts = list(map(float, line.split()))
        return parts[2:]  # Skip dimensions

    def init_alpha(self):
        # Initialize alpha for t=1
        self.alphas.append([
            self.initial_state[i] * self.emission_matrix[i][self.emissions[0]]
            for i in range(len(self.initial_state))
        ])
    
    def forward_algorithm(self):
        # Compute alpha values iteratively for t=2 to T
        for t in range(1, len(self.emissions)):
            alpha_t = []
            for i in range(len(self.transition_matrix)):
                # Compute alpha_t(i)
                alpha_t_i = sum(
                    self.alphas[t-1][j] * self.transition_matrix[j][i]
                    for j in range(len(self.transition_matrix))
                ) * self.emission_matrix[i][self.emissions[t]]
                alpha_t.append(alpha_t_i)
            self.alphas.append(alpha_t)

        # Compute the final probability P(O1:T)
        return sum(self.alphas[-1])

def main():
    hmm = HMM()
    hmm.read_input()
    hmm.init_alpha()
    result = hmm.forward_algorithm()
    print(result)

if __name__ == "__main__":
    main()
