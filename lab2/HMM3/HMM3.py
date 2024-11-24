import sys
from math import log

class HMM:
    def __init__(self):
        self.transition_matrix = []
        self.emission_matrix = []
        self.initial_state = []
        self.emissions = []
        
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
        
    def forward_pass(self, emissions):
        N = len(self.transition_matrix)
        T = len(emissions)
        alpha = [[0.0] * N for _ in range(T)]
        scale = [0.0] * T
        
        # Initialize first time step
        for i in range(N):
            alpha[0][i] = self.initial_state[i] * self.emission_matrix[i][emissions[0]]
        scale[0] = sum(alpha[0])
        for i in range(N):
            alpha[0][i] /= scale[0]
        
        # Forward pass
        for t in range(1, T):
            for j in range(N):
                sum_alpha = 0.0
                for i in range(N):
                    sum_alpha += alpha[t-1][i] * self.transition_matrix[i][j]
                alpha[t][j] = sum_alpha * self.emission_matrix[j][emissions[t]]
            
            scale[t] = sum(alpha[t])
            for j in range(N):
                alpha[t][j] /= scale[t]
            
        return alpha, scale
        
    def backward_pass(self, emissions, scale):
        N = len(self.transition_matrix)
        T = len(emissions)
        beta = [[0.0] * N for _ in range(T)]
        
        # Initialize last time step
        for i in range(N):
            beta[T-1][i] = 1.0 / scale[T-1]
        
        # Backward pass
        for t in range(T-2, -1, -1):
            for i in range(N):
                sum_beta = 0.0
                for j in range(N):
                    sum_beta += (self.transition_matrix[i][j] * 
                               self.emission_matrix[j][emissions[t+1]] * 
                               beta[t+1][j])
                beta[t][i] = sum_beta / scale[t]
            
        return beta
    
    def compute_di_gamma(self, t, alpha, beta, emission):
        N = len(self.transition_matrix)
        xi = [[0.0] * N for _ in range(N)]
        sum_xi = 0.0
        
        for i in range(N):
            for j in range(N):
                xi[i][j] = (alpha[t][i] * 
                           self.transition_matrix[i][j] * 
                           self.emission_matrix[j][emission] * 
                           beta[t+1][j])
                sum_xi += xi[i][j]
                
        # Normalize xi
        for i in range(N):
            for j in range(N):
                xi[i][j] /= sum_xi if sum_xi > 0 else 1.0
                
        return xi
    
    def compute_gamma(self, t, alpha, beta):
        N = len(self.transition_matrix)
        gamma = [0.0] * N
        sum_gamma = 0.0
        
        for i in range(N):
            gamma[i] = alpha[t][i] * beta[t][i]
            sum_gamma += gamma[i]
            
        # Normalize gamma
        for i in range(N):
            gamma[i] /= sum_gamma if sum_gamma > 0 else 1.0
            
        return gamma
    
    def baum_welch(self, max_iter=100, eps=1e-6):
        old_log_prob = float('-inf')
        
        for iteration in range(max_iter):
            # E-step
            alpha, scale = self.forward_pass(self.emissions)
            beta = self.backward_pass(self.emissions, scale)
            
            # Compute log probability
            log_prob = sum(log(s) if s > 0 else float('-inf') for s in scale)
            if log_prob - old_log_prob < eps and iteration > 0:
                break
            old_log_prob = log_prob
            
            T = len(self.emissions)
            N = len(self.transition_matrix)
            M = len(self.emission_matrix[0])
            
            # M-step
            # Update transition matrix
            new_transition = [[0.0] * N for _ in range(N)]
            for i in range(N):
                denominator = 0.0
                for t in range(T-1):
                    gamma_t = self.compute_gamma(t, alpha, beta)
                    denominator += gamma_t[i]
                
                for j in range(N):
                    numerator = 0.0
                    for t in range(T-1):
                        xi = self.compute_di_gamma(t, alpha, beta, self.emissions[t+1])
                        numerator += xi[i][j]
                    new_transition[i][j] = numerator / denominator if denominator > 0 else 0.0
            
            # Update emission matrix
            new_emission = [[0.0] * M for _ in range(N)]
            for i in range(N):
                denominator = 0.0
                for t in range(T):
                    gamma_t = self.compute_gamma(t, alpha, beta)
                    denominator += gamma_t[i]
                
                for k in range(M):
                    numerator = 0.0
                    for t in range(T):
                        if self.emissions[t] == k:
                            gamma_t = self.compute_gamma(t, alpha, beta)
                            numerator += gamma_t[i]
                    new_emission[i][k] = numerator / denominator if denominator > 0 else 0.0
            
            self.transition_matrix = new_transition
            self.emission_matrix = new_emission

    def format_matrix(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        result = f"{rows} {cols}"
        for row in matrix:
            for val in row:
                result += f" {val}"
        return result

def main():
    hmm = HMM()
    hmm.read_input()
    hmm.baum_welch()
    
    # Print results in required format
    print(hmm.format_matrix(hmm.transition_matrix))
    print(hmm.format_matrix(hmm.emission_matrix))

if __name__ == "__main__":
    main()