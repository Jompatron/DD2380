#!/usr/bin/env python3
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
from time import time
TIME_THRESHOLD = 75*1e-3

Lvl1_scores = [] # List with scores for layer 1
Lvl1_nodes = [] # List with nodes for layer 1

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

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate game tree object
        first_msg = self.receiver()
        # Initialize your minimax model
        #model = self.initialize_model(initial_data=first_msg)

        while True:
            msg = self.receiver()
            self.start_time = time()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)


            # Possible next moves: "stay", "left", "right", "up", "down"

            score, best_move = self.searching(node = node)

            global Lvl1_nodes
            global Lvl1_scores
            Lvl1_scores = []
            Lvl1_nodes = []

            self.sender({"action": best_move})


    # searching down a given max depth
    def searching(self, node):
        num_iterations = 10
        
        global Lvl1_nodes
        global Lvl1_scores

        for i in range(1, num_interation):
            score, move, timeout = self.next_best_move(currentNode = node, depth = i, alpha = -1000, beta = 1000, player = 0, maxDepth = i)
            
            Lvl1_nodes = self.sorting(Lvl1_nodes, Lvl1_scores)
            
            if timeout:
                break

            Lvl1_nodes = self.sorting(Lvl1_nodes, Lvl1_scores)

            Lvl1_scores = []

        return score, move

    def sorting(self, list1, list2):
        # Orders children by their highest score value from left to right
        keydict = dict(zip(list1, list2)) # We only get last 5 of list to avoid the 5 scores that get appened from the first iteration which we don't need
        list1.sort(key=keydict.get, reverse = True)

        return list1


    def next_best_move(self, currentNode, depth, alpha, beta, player, maxDepth):

        global Lvl1_nodes
        global Lvl1_scores

        if time() - self.start_time > (TIME_THRESHOLD - 0.05):
            timeout = True
            #print("Timeout at depth:", maxDepth)
            evaluation = self.heuristic(currentNode)
            
    def heuristic(self, currentNode):
        
        return evaluation
