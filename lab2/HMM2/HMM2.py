import sys

class HMM:
    def __init__(self):
        self.transition_matrix = []
        self.emission_matrix = []
        self.initial_state = []
        self.emissions = []
        self.delta_vector = []

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

    def init_delta(self):
        # Initialize delta for t=1
        self.delta_vector.append([
            self.initial_state[i] * self.emission_matrix[i][self.emissions[0]]
            for i in range(len(self.initial_state))
        ])

    def print_path(self, backpointers, state):
        if state == 0:
            return
        self.print_path(backpointers, backpointers[state])
        print(state, end=' ')

    def argmax(self, values):
        max_value = max(values)
        return values.index(max_value)
    
    def print_path(self, backpointers, state):
        print_str = str(state) + ' '
        for backpointers_t in reversed(backpointers):
            state = backpointers_t[state]
            print_str += str(state) + ' '
        print(print_str[::-1].strip())
    
    def viterbi_algorithm(self):
        backpointers = [] # Store the backpointers for each state

        # Compute delta values iteratively for t=2 to T
        for t in range(1, len(self.emissions)):
            delta_t = []
            backpointers_t = []
            for i in range(len(self.transition_matrix)):
                # Compute delta_t(i)
                delta_t_i = max(
                    self.delta_vector[t-1][j] * self.transition_matrix[j][i]
                    for j in range(len(self.transition_matrix))
                ) * self.emission_matrix[i][self.emissions[t]]
                
                delta_t.append(delta_t_i)

                # Store the backpointer for the most probable path
                backpointer = self.argmax(
                    [self.delta_vector[t-1][j] * self.transition_matrix[j][i]
                    for j in range(len(self.transition_matrix))]
                )
                backpointers_t.append(backpointer)
            backpointers.append(backpointers_t)
            
            self.delta_vector.append(delta_t)

        # Return the most probable path
        print(self.delta_vector)
        print(backpointers)
        self.print_path(backpointers, self.argmax(self.delta_vector[-1]))

def main():
    hmm = HMM()
    hmm.read_input()
    hmm.init_delta()
    hmm.viterbi_algorithm()

if __name__ == "__main__":
    main()
