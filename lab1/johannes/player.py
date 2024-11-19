#!/usr/bin/env python3
# import random
# import math
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
        self.max_depth_limit = 8
        self.transposition_table = {}
        self.start_time = None
        self.time_limit = None
        self.total_time_limit = 0.095  # 95ms to stay within 100ms limit

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
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        best_move = None
        best_score = float("-inf")
        self.start_time = time.time()

        # Iterative Deepening
        for depth in range(1, self.max_depth_limit + 1):
            self.depth_limit = depth
            self.transposition_table = {}  # Clear transposition table at each depth
            try:
                # Adjust time limit for this iteration
                elapsed_time = time.time() - self.start_time
                self.time_limit = self.total_time_limit - elapsed_time
                if self.time_limit <= 0:
                    break
                # Do minimax search with current depth limit
                curr_move, curr_score = self.minimax(
                    initial_tree_node, depth=0, alpha=float("-inf"), beta=float("inf"), max_player=True
                )
                if curr_score > best_score:
                    best_score = curr_score
                    best_move = curr_move
            except TimeoutError:
                break  # Exit if time is up

        return ACTION_TO_STR[best_move] if best_move is not None else "stay"

    def minimax(self, node: Node, depth: int, alpha: float, beta: float, max_player: bool):
        # Check if time limit has been exceeded
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError

        # Transposition table lookup
        state_key = self.get_state_key(node)
        if state_key in self.transposition_table:
            stored_depth, stored_eval = self.transposition_table[state_key]
            if stored_depth >= self.depth_limit - depth:
                return None, stored_eval

        # Check for terminal state or depth limit
        if depth >= self.depth_limit or self.is_terminal(node):
            eval_score = self.evaluate(node)
            return None, eval_score

        if max_player:
            max_eval = float("-inf")
            best_move = None
            children = self.get_ordered_children(node, max_player=True)
            for child in children:
                _, eval = self.minimax(child, depth + 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = child.move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            # Store in transposition table
            self.transposition_table[state_key] = (self.depth_limit - depth, max_eval)
            return best_move, max_eval
        else:
            min_eval = float("inf")
            best_move = None
            children = self.get_ordered_children(node, max_player=False)
            for child in children:
                _, eval = self.minimax(child, depth + 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = child.move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            # Store in transposition table
            self.transposition_table[state_key] = (self.depth_limit - depth, min_eval)
            return best_move, min_eval

    def get_ordered_children(self, node: Node, max_player: bool):
        """
        Orders children nodes based on heuristic to improve alpha-beta pruning.
        """
        children = node.compute_and_get_children()
        # Evaluate each child node
        child_evals = []
        for child in children:
            eval = self.evaluate(child)
            child_evals.append((eval, child))

        # Sort based on evaluation score
        child_evals.sort(key=lambda x: x[0], reverse=max_player)
        # Return sorted children
        return [child for _, child in child_evals]

    def get_state_key(self, node: Node):
        # Use positions of fish, positions of hooks, and scores
        fish_positions = node.state.get_fish_positions()
        hook_positions = node.state.get_hook_positions()
        player_scores = node.state.get_player_scores()

        # Create tuples
        fish_positions_tuple = tuple(sorted(fish_positions.items()))
        hook_positions_tuple = tuple(hook_positions)
        player_scores_tuple = tuple(player_scores)

        state_key = (fish_positions_tuple, hook_positions_tuple, player_scores_tuple)

        return state_key

    def is_terminal(self, node: Node) -> bool:
        # Terminal if no more fish
        return len(node.state.get_fish_positions()) == 0

    def evaluate(self, node: Node) -> float:
        scores = node.state.get_player_scores()
        our_score = scores[0]
        opponent_score = scores[1]
        fish_positions = node.state.get_fish_positions()
        our_hook_pos = node.state.get_hook_positions()[0]
        opponent_hook_pos = node.state.get_hook_positions()[1]
        fish_scores = node.state.get_fish_scores()

        # Base score is the difference in scores
        score = (our_score - opponent_score) * 100

        # Add evaluation for each fish
        for fish_id, fish_pos in fish_positions.items():
            fish_value = fish_scores[fish_id]

            # Distance from our hook to the fish
            our_distance = self.manhattan_distance(our_hook_pos, fish_pos)
            # Distance from opponent's hook to the fish
            opponent_distance = self.manhattan_distance(opponent_hook_pos, fish_pos)

            # If we are closer to the fish
            if our_distance < opponent_distance:
                score += fish_value / (our_distance + 1)
            elif our_distance > opponent_distance:
                score -= fish_value / (opponent_distance + 1)
            else:
                # If equal distance, consider who can get there first
                score += fish_value / (our_distance + 2)

        return score

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
