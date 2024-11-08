#!/usr/bin/env python3
#import random
#import math
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.maxDepthLimit = 8

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node: Node) -> str:
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        bestMove = None
        bestScore = float("-inf")
        startTime = time.time()
        timeLimit = 75*1e-3 / 1.5 # add some margin

        # iterative deepening search
        for depth in range(1, self.maxDepthLimit+1):
            self.depth_limit = depth
            try:
                # do minimax search with current depth limit
                currMove, currScore = self.minimax(initial_tree_node, depth=0, maxPlayer=True, startTime=startTime, timeLimit=timeLimit)
                if currScore > bestScore:
                    bestScore = currScore
                    bestMove = currMove

            except TimeoutError:
                break # exit if time is up

        return ACTION_TO_STR[bestMove] if bestMove is not None else "stay"

    def minimax(self, node: Node, depth: int, maxPlayer: bool, alpha=float("-inf"), beta=float("inf"), startTime=None, timeLimit=None):
        # check if time limit has been exceeded
        timeDiff = time.time() - startTime
        #print("Curr time:", timeDiff)
        
        # check depth limit and terminal state
        if depth == self.depth_limit or self.is_terminal(node) or timeDiff > timeLimit: # add some margin for time limit
            return None, self.evaluate(node)
        
        if maxPlayer:
            max_eval = float("-inf")
            best_move = None
            children = node.compute_and_get_children()
            for child in children:
                _, eval = self.minimax(child, depth + 1, False, alpha, beta, startTime, timeLimit)

                if eval > max_eval:
                    max_eval = eval
                    best_move = child.move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return best_move, max_eval

        else:
            min_eval = float("inf")
            best_move = None
            children = node.compute_and_get_children()
            for child in children:
                _, eval = self.minimax(child, depth + 1, True, alpha, beta, startTime, timeLimit)

                if eval < min_eval:
                    min_eval = eval
                    best_move = child.move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return best_move, min_eval

    # If no more fish from this state, it's terminal state
    def is_terminal(self, node: Node) -> bool:
        return len(node.state.get_fish_positions()) == 0
    
    def evaluate(self, node: Node) -> int:
        scores = node.state.get_player_scores()
        max_score = scores[0]
        min_score = scores[1]
        fish_positions = node.state.get_fish_positions()
        maxPosition = node.state.get_hook_positions()[0] # our position
        minPosition = node.state.get_hook_positions()[1]
        fish_scores = node.state.get_fish_scores()

        # if no fish are left, return the score difference
        if not fish_positions:
            return max_score - min_score

        # Calculate a weighted score based on the distance to each fish and its value
        score = max_score - min_score
        for fish_id, fish_pos in fish_positions.items():
            fish_value = fish_scores.get(fish_id)

            # if on the other side of the screen (on the opponents side)
            if minPosition[0] > maxPosition[0] and fish_pos[0] >= minPosition[0]:
                xDist = maxPosition[0] + (20 - fish_pos[0])
            else: # if on our side
                xDist = maxPosition[0] - fish_pos[0]
            yDist = maxPosition[1] - fish_pos[1]
            distance = self.hypot(xDist, yDist)
            
            # add weighted fish score inversely proportional to the distance
            score += fish_value / (distance + 1)  # +1 to avoid division by zero
        
        #print("Evaluating Node - Player 0 score:", max_score, "Player 1 score:", min_score, "Evaluation score:", score)
        return score

    def hypot(self, x, y):
        return (x**2 + y**2)**(1/2)




