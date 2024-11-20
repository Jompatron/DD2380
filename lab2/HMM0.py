import sys

class HMM:
    def __init__(self):
        self.transition_matrix = []
        self.emission_matrix = []
        self.initial_state = []

    def read_input(self):
        data = sys.stdin.read().strip().split('\n')
        self.transition_matrix = self.read_matrix(data[0])
        self.emission_matrix = self.read_matrix(data[1])
        self.initial_state = self.read_vector(data[2])

    def read_matrix(self, line):
        parts = list(map(float, line.split()))
        rows, cols = int(parts[0]), int(parts[1])
        values = parts[2:]
        return [values[i * cols:(i + 1) * cols] for i in range(rows)]

    def read_vector(self, line):
        parts = list(map(float, line.split()))
        return parts[2:]  # Skip dimensions

    def calculate_emission_probability(self):
        # Step 1: Compute intermediate state (π * A)
        intermediate = [
            sum(self.initial_state[row] * self.transition_matrix[row][col]
                for row in range(len(self.transition_matrix)))
            for col in range(len(self.transition_matrix[0]))
        ]

        # Step 2: Compute final emission probabilities (intermediate * B)
        final_state = [
            sum(intermediate[row] * self.emission_matrix[row][col]
                for row in range(len(self.emission_matrix)))
            for col in range(len(self.emission_matrix[0]))
        ]

        return [final_state]

    def write_output(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        result = f"{rows} {cols} " + " ".join(map(str, matrix[0]))
        print(result)

def main():
    hmm = HMM()
    hmm.read_input()
    emission_probs = hmm.calculate_emission_probability()
    hmm.write_output(emission_probs)

if __name__ == "__main__":
    main()
