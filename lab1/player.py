#!/usr/bin/env python3
#!/usr/bin/env python3
import time
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        while True:
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.startTime = None
        self.timeLimit = None
        self.depth_limit = None
        self.transposition_table = {}  # Cache for evaluated nodes

    def player_loop(self):
        first_msg = self.receiver()

        while True:
            msg = self.receiver()
            node = Node(message=msg, player=0)
            best_move = self.search_best_next_move(initial_tree_node=node)
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node: Node) -> str:
        bestMove = None
        self.startTime = time.time()
        totalTimeLimit = 75 * 1e-3  # Total time per move
        self.timeLimit = totalTimeLimit / 1.5  # Add margin

        depth = 1
        while True:
            self.depth_limit = depth
            self.transposition_table = {}  # Clear cache for new depth
            try:
                currMove, currScore = self.minimax(
                    node=initial_tree_node, depth=0, maxPlayer=True, alpha=float("-inf"), beta=float("inf")
                )
                if currMove is not None:
                    bestMove = currMove
                depth += 1
            except TimeoutError:
                break  # Exit if time is up

        return ACTION_TO_STR[bestMove] if bestMove is not None else "stay"

    def minimax(self, node: Node, depth: int, maxPlayer: bool, alpha: float, beta: float):
        # Time check
        if time.time() - self.startTime > self.timeLimit:
            raise TimeoutError

        # Transposition table lookup
        node_hash = hash(node.state)
        if node_hash in self.transposition_table and self.transposition_table[node_hash][0] >= depth:
            return self.transposition_table[node_hash][1]

        # Terminal or depth limit check
        if depth == self.depth_limit or self.is_terminal(node):
            eval_score = self.evaluate(node)
            self.transposition_table[node_hash] = (depth, (None, eval_score))
            return None, eval_score

        if maxPlayer:
            max_eval = float("-inf")
            best_move = None
            children = node.compute_and_get_children()

            # Move ordering
            children.sort(key=lambda child: self.quick_evaluate(child), reverse=True)

            for child in children:
                _, eval = self.minimax(child, depth + 1, False, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = child.move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff

            # Store in transposition table
            self.transposition_table[node_hash] = (depth, (best_move, max_eval))
            return best_move, max_eval

        else:
            min_eval = float("inf")
            best_move = None
            children = node.compute_and_get_children()

            # Move ordering
            children.sort(key=lambda child: self.quick_evaluate(child))

            for child in children:
                _, eval = self.minimax(child, depth + 1, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = child.move
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff

            # Store in transposition table
            self.transposition_table[node_hash] = (depth, (best_move, min_eval))
            return best_move, min_eval

    def is_terminal(self, node: Node) -> bool:
        return len(node.state.get_fish_positions()) == 0
    

    def quick_evaluate(self, node: Node) -> float:
        scores = node.state.get_player_scores()
        our_score = scores[0]
        opp_score = scores[1]
        fish_positions = node.state.get_fish_positions()
        our_pos = node.state.get_hook_positions()[0]
        fish_scores = node.state.get_fish_scores()

        # Calculate the potential gain from the closest fish
        min_distance = float("inf")
        potential_gain = 0
        for fish_id, fish_pos in fish_positions.items():
            fish_value = fish_scores.get(fish_id)
            distance = self.calculate_distance(our_pos, fish_pos, 20)
            if distance < min_distance:
                min_distance = distance
                potential_gain = fish_value / (distance + 1)

        return (our_score - opp_score) * 10 + potential_gain


# In your PlayerControllerMinimax class

    def evaluate(self, node: Node) -> float:
        scores = node.state.get_player_scores()
        our_score = scores[0]
        opp_score = scores[1]
        fish_positions = node.state.get_fish_positions()
        our_pos = node.state.get_hook_positions()[0]
        opp_pos = node.state.get_hook_positions()[1]
        fish_scores = node.state.get_fish_scores()

        if not fish_positions:
            return our_score - opp_score

        TOTAL_WIDTH = 20
        heuristic_score = (our_score - opp_score) * 10  # Amplify score differences

        for fish_id, fish_pos in fish_positions.items():
            fish_value = fish_scores.get(fish_id)

            # Calculate distances to the fish for both players
            our_distance = self.calculate_distance(our_pos, fish_pos, TOTAL_WIDTH)
            opp_distance = self.calculate_distance(opp_pos, fish_pos, TOTAL_WIDTH)

            # Adjust heuristic based on potential to catch the fish
            our_potential = fish_value / (our_distance + 1)
            opp_potential = fish_value / (opp_distance + 1)
            heuristic_score += our_potential - opp_potential

        return heuristic_score

    def calculate_distance(self, pos1, pos2, width):
        xDist = min(abs(pos1[0] - pos2[0]), width - abs(pos1[0] - pos2[0]))
        yDist = abs(pos1[1] - pos2[1])
        return (xDist ** 2 + yDist ** 2) ** 0.5
