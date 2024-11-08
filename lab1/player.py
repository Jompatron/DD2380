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

        while True:
            msg = self.receiver()
            self.start_time = time()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)


            # Possible next moves: "stay", "left", "right", "up", "down"


            score, best_move = self.search(num_interation = 10, node = node)

            global Lvl1_nodes
            global Lvl1_scores
            Lvl1_scores = []
            Lvl1_nodes = []

            self.sender({"action": best_move, "search_time": None})


    # search with max depth given
    def search(self, num_interation, node):

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
            if currentNode.move is not None:
                next_move = currentNode.move
            else:
                next_move = 0
            return evaluation, ACTION_TO_STR[next_move], timeout
        
        children = currentNode.compute_and_get_children()

        if depth == 0:
            evaluation = self.heuristic(currentNode)
            timeout = False
            return evaluation, ACTION_TO_STR[currentNode.move], timeout

        if player == 0:
            maxVal = -1000 # Want to maximise, give worst possible value to start
            next_move = -1
            for child in children:
                childVal, returnMove, timeout = self.next_best_move(child, depth-1, alpha, beta, 1, maxDepth)
                if depth == maxDepth: # If at root node
                    Lvl1_scores.append(childVal)
                if childVal > maxVal:
                    next_move = returnMove
                    maxVal = childVal
                alpha = max(alpha, maxVal)
                if beta <= alpha:
                    break
            return maxVal, next_move, timeout
        else:
            minVal = 1000 # Want to minimise, give worst possible value to start
            next_move = -1
            for child in children:
                childVal, returnMove, timeout = self.next_best_move(child, depth-1, alpha, beta, 0, maxDepth)
                if childVal < minVal:
                    next_move = returnMove
                    minVal = childVal
                beta = min(beta, minVal)
                if beta <= alpha:
                    break
            return minVal, next_move, timeout


    def heuristic(self, currentNode):

        # Current player scores --> (0, 10)
        playerScore =  currentNode.state.get_player_scores()
        # Fish positions --> {0: (6, 16), 1: (1, 14), 3: (8, 13), 4: (19, 6)}
        fishPos = currentNode.state.get_fish_positions()
        # Hook positions --> {0: (6, 12), 1: (11, 16)}
        hookPos = currentNode.state.get_hook_positions()
        # Fish scores --> {0: 11, 1: 2, 2: 10, 3: 2, 4: 11}
        fishScore = currentNode.state.get_fish_scores()


        fishPos_keys = fishPos.keys()
        
        fishDis0 = {}   # contains all fish ditances to hook of player 0
        fishDis1 = {}   # contains all fish ditances to hook of player 1
        fishPoints = {} # contains score of remaining fish

        evaluation = 0

        closestDistance0 = 0
        
        playerFishScore0 = 0
        playerFishScore1 = 0
        
        
        if fishPos:
            for key in fishPos_keys:
                #if fishScore[key] > 0:
                x_dis_zero=min(abs(fishPos[key][0] - hookPos[0][0]), 20-abs(fishPos[key][0] - hookPos[0][0]))
                x_dis_one=min(abs(fishPos[key][0] - hookPos[1][0]), 20-abs(fishPos[key][0] - hookPos[1][0]))
                y_dis_zero = fishPos[key][1]-hookPos[0][1]
                y_dis_one = fishPos[key][1]-hookPos[1][1]
                fishDis0[key] = ((x_dis_zero)**2 + (y_dis_zero)**2)**0.5
                fishDis1[key] = ((x_dis_one)**2 + (y_dis_one)**2)**0.5
                fishPoints[key] = fishScore[key]

           
            # Make sure fishDis is populated, aka nonzero score fish are remaining:
            if fishDis0:
                closestDistance0 = min(fishDis0.values())
                
                playerFishScore0 = sum({key: fishPoints[key] / (fishDis0.get(key, 0) + 0.01) for key in fishDis0 if key in fishPoints}.values())
                playerFishScore1 = sum({key: fishPoints[key] / (fishDis1.get(key, 0) + 0.01) for key in fishDis1 if key in fishPoints}.values())


        evaluation = 0.55 * (playerScore[0] - playerScore[1]) + 0.5 *(playerFishScore0 - playerFishScore1) - closestDistance0
        
        return evaluation
