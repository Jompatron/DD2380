#!/usr/bin/env python3
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
from time import time

# Time threshold for move calculation to avoid timeout
TIME_THRESHOLD = 75 * 1e-3  

# Global lists for scores and nodes at the first level of the game tree
Lvl1_scores = []  # Holds scores for root level nodes
Lvl1_nodes = []   # Holds nodes for root level nodes

class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Loop function for a human player to interact with the game.
        In each iteration, the player makes a move through keyboard input, 
        which is sent to the game. Then it receives an update on the game 
        state and computes the next move.
        """
        while True:
            # Wait for game message
            msg = self.receiver()
            if msg["game_over"]:
                return  # Exit if the game is over


class PlayerControllerMinimax(PlayerController):
    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main game loop for the AI player using minimax with alpha-beta pruning.
        """
        first_msg = self.receiver()  # Get initial game message

        while True:
            # Receive updated game state
            msg = self.receiver()
            self.start_time = time()  # Record start time to monitor time limit

            # Create root node of the game tree based on the current game state
            node = Node(message=msg, player=0)

            # Use minimax search with iterative deepening to determine the best move
            score, best_move = self.search(num_interation=10, node=node)

            # Reset global variables for level 1 after each move
            global Lvl1_nodes, Lvl1_scores
            Lvl1_scores = []
            Lvl1_nodes = []

            # Send the chosen move to the game
            self.sender({"action": best_move, "search_time": None})

    def search(self, num_interation, node):
        """
        Perform iterative deepening search up to the given depth.
        """
        global Lvl1_nodes, Lvl1_scores

        # Perform search for each depth level
        for i in range(1, num_interation):
            # Run minimax with alpha-beta pruning
            score, move, timeout = self.next_best_move(
                currentNode=node, depth=i, alpha=-1000, beta=1000, player=0, maxDepth=i
            )
            
            # Sort level 1 nodes based on their scores
            Lvl1_nodes = self.sorting(Lvl1_nodes, Lvl1_scores)
            
            # Stop if time threshold is exceeded
            if timeout:
                break

            # Clear scores for next iteration
            Lvl1_scores = []

        return score, move

    def sorting(self, list1, list2):
        """
        Sorts the list of nodes based on their scores in descending order.
        """
        # Create dictionary to map nodes to their scores
        keydict = dict(zip(list1, list2))
        # Sort list1 (nodes) by their scores from list2 in descending order
        list1.sort(key=keydict.get, reverse=True)
        return list1

    def next_best_move(self, currentNode, depth, alpha, beta, player, maxDepth):
        """
        Minimax with alpha-beta pruning to calculate the best next move.
        """
        global Lvl1_nodes, Lvl1_scores

        # Check if time threshold is exceeded
        if time() - self.start_time > (TIME_THRESHOLD - 0.05):
            timeout = True
            # Calculate heuristic evaluation if timeout occurs
            evaluation = self.heuristic(currentNode)
            next_move = currentNode.move if currentNode.move is not None else 0
            return evaluation, ACTION_TO_STR[next_move], timeout
        
        # Generate children nodes of the current node
        children = currentNode.compute_and_get_children()

        # Perform heuristic evaluation if at maximum depth
        if depth == 0:
            evaluation = self.heuristic(currentNode)
            return evaluation, ACTION_TO_STR[currentNode.move], False

        # Maximize or minimize based on the player's turn
        if player == 0:
            # Player 0 is maximizing
            maxVal = -1000
            next_move = -1
            for child in children:
                # Recursively calculate the child node value
                childVal, returnMove, timeout = self.next_best_move(child, depth-1, alpha, beta, 1, maxDepth)
                if depth == maxDepth:
                    Lvl1_scores.append(childVal)  # Track scores at root level
                if childVal > maxVal:
                    maxVal = childVal
                    next_move = returnMove
                alpha = max(alpha, maxVal)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return maxVal, next_move, timeout
        else:
            # Player 1 is minimizing
            minVal = 1000
            next_move = -1
            for child in children:
                # Recursively calculate the child node value
                childVal, returnMove, timeout = self.next_best_move(child, depth-1, alpha, beta, 0, maxDepth)
                if childVal < minVal:
                    minVal = childVal
                    next_move = returnMove
                beta = min(beta, minVal)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return minVal, next_move, timeout

    def heuristic(self, currentNode):
        """
        Calculate the heuristic evaluation for the current node.
        """
        # Get the scores, fish positions, hook positions, and fish scores
        playerScore = currentNode.state.get_player_scores()
        fishPos = currentNode.state.get_fish_positions()
        hookPos = currentNode.state.get_hook_positions()
        fishScore = currentNode.state.get_fish_scores()

        # Initialize dictionaries to store distances and scores
        fishDis0 = {}  # Fish distances to hook of player 0
        fishDis1 = {}  # Fish distances to hook of player 1
        fishPoints = {}  # Scores of remaining fish

        # Initialize scores and evaluation
        evaluation = 0
        closestDistance0 = 0
        playerFishScore0 = 0
        playerFishScore1 = 0
        
        # Calculate distances and scores for each fish
        for key in fishPos.keys():
            x_dis_zero = min(abs(fishPos[key][0] - hookPos[0][0]), 20 - abs(fishPos[key][0] - hookPos[0][0]))
            x_dis_one = min(abs(fishPos[key][0] - hookPos[1][0]), 20 - abs(fishPos[key][0] - hookPos[1][0]))
            y_dis_zero = fishPos[key][1] - hookPos[0][1]
            y_dis_one = fishPos[key][1] - hookPos[1][1]
            # Calculate Euclidean distance
            fishDis0[key] = (x_dis_zero**2 + y_dis_zero**2) ** 0.5
            fishDis1[key] = (x_dis_one**2 + y_dis_one**2) ** 0.5
            fishPoints[key] = fishScore[key]

        # Calculate proximity-based scores if there are fish on the board
        if fishDis0:
            closestDistance0 = min(fishDis0.values())  # Closest fish to player 0's hook
            # Calculate score based on fish proximity to each player
            playerFishScore0 = sum(fishPoints[key] / (fishDis0[key] + 0.01) for key in fishDis0)
            playerFishScore1 = sum(fishPoints[key] / (fishDis1[key] + 0.01) for key in fishDis1)

        # Combine player scores, fish proximity scores, and closest distance into final evaluation
        evaluation = 0.55 * (playerScore[0] - playerScore[1]) + 0.5 * (playerFishScore0 - playerFishScore1) - closestDistance0
        return evaluation
