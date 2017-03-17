"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import operator

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

open_book_level1 = [(2,2), (2,3), (2,4), (3,2), (3,3), (3,4)]
open_book_level2 = [(1,2), (1,3), (1,4), (2,1), (2, 5, (3, 1), (3,5), (4,1), (4,5), (5,2), (5,3), (5,4))]
corner_spots = [(0,0), (0,6), (6,0), (6,6)]
only_three_options = [(0,1),(0,5), (1,0), (1,6), (5,0), (5,6), (6,1), (6,5)]

better_moves = open_book_level1 + open_book_level2
worse_moves = corner_spots + only_three_options

def pick_good_move(legal_moves):
    """ Check if we have a good move in the set of legal moves.
        pick one that is best suitable or return a random one if non
        exist in the open book

    Parameters
    ----------
    legal_moves: a list of legal moves

    returns
    tuple
        picked from the list

    """
    if not legal_moves:
        return ()

    # check open_book_level1 which gives 8 possible next moves
    to_choose_from = list(set(open_book_level1) & set(legal_moves))
    # if not to_choose_from:
    #     # check level2 which gives 6 possible next moves
    #     to_choose_from = list(set(open_book_level2) & set(legal_moves))

    if to_choose_from:
        return to_choose_from[random.randint(0, len(to_choose_from) -1)]

    return legal_moves[random.randint(0, len(legal_moves) - 1)]

def jaccard_score_v2(game, player):
    """ This score  compare the jaccard similarity of the player legal moves and
        the better_moves to the the opponent player jaccard similarity

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    jaccard_score = 0;
    player_legal_moves = game.get_legal_moves(player)
    opp_player_legal_moves = game.get_legal_moves(game.get_opponent(player))

    num_intersect_legal_moves = len(list(set(player_legal_moves) & set(better_moves)))
    num_union_legal_moves = len(player_legal_moves) + len(better_moves)
    player_jaccard_index = num_intersect_legal_moves / float(num_union_legal_moves)

    num_intersect_legal_moves = len(list(set(opp_player_legal_moves) & set(better_moves)))
    num_union_legal_moves = len(opp_player_legal_moves) + len(better_moves)
    opp_player_jaccard_index = num_intersect_legal_moves / float(num_union_legal_moves)

    jaccard_score = player_jaccard_index - opp_player_jaccard_index
    if jaccard_score > 0:
        return  jaccard_score

    own_moves = len(player_legal_moves)
    opp_moves = len(opp_player_legal_moves)
    return float(own_moves - opp_moves)

def jaccard_score(game, player):
    """ This score measure the jaccard similarity of the player legal moves to
        the opponent player legal moves

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    jaccard_index = 0;
    player_legal_moves = game.get_legal_moves(player)
    opp_player_legal_moves = game.get_legal_moves(game.get_opponent(player))

    num_intersect_legal_moves = len(list(set(player_legal_moves) & set(opp_player_legal_moves)))
    num_union_legal_moves = len(player_legal_moves) + len(opp_player_legal_moves)
    jaccard_index = num_intersect_legal_moves / float(num_union_legal_moves)

    if jaccard_index > 0:
        return  jaccard_index

    own_moves = len(player_legal_moves)
    opp_moves = len(opp_player_legal_moves)
    return float(own_moves - opp_moves)

def negative_impact_score(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players. What we add is a bonus if the own_moves set is in open_book_level1 or open_book_level2

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    player_legal_moves = game.get_legal_moves(player)
    opp_player_legal_moves = game.get_legal_moves(game.get_opponent(player))

    impact = 0;
    # check against the wrose locations
    # if game.move_count > 25:
    player_worse_moves =  len(list(set(corner_spots) & set(player_legal_moves)))
    opp_player_worse_moves = len(list(set(corner_spots) & set(opp_player_legal_moves)))
    impact = opp_player_worse_moves - player_worse_moves

    own_moves = len(player_legal_moves)
    opp_moves = len(opp_player_legal_moves)
    return float(own_moves - opp_moves + impact)

def positive_impact_score(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players. What we add is a bonus if the own_moves set is in open_book_level1 or open_book_level2

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    reward = 0;
    player_legal_moves = game.get_legal_moves(player)
    opp_player_legal_moves = game.get_legal_moves(game.get_opponent(player))

    if game.move_count <= 20:
        player_better_moves = len(list(set(better_moves) & set(player_legal_moves)))
        opp_player_better_moves = len(list(set(better_moves) & set(opp_player_legal_moves)))
        reward = (player_better_moves - opp_player_better_moves)

    own_moves = len(player_legal_moves)
    opp_moves = len(opp_player_legal_moves)
    return float(own_moves - opp_moves + reward)


def impact_score(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players. What we add is a bonus if the own_moves set is in open_book_level1 or open_book_level2

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    reward = 0;
    penalty = 0;

    player_legal_moves = game.get_legal_moves(player)
    opp_player_legal_moves = game.get_legal_moves(game.get_opponent(player))

    if game.move_count <= 20:
        player_better_moves = len(list(set(better_moves) & set(player_legal_moves)))
        opp_player_better_moves = len(list(set(better_moves) & set(opp_player_legal_moves)))
        reward = (player_better_moves - opp_player_better_moves)
    else:
        # check against the wrose locations
        player_worse_moves =  len(list(set(worse_moves) & set(player_legal_moves)))
        opp_player_worse_moves = len(list(set(worse_moves) & set(opp_player_legal_moves)))
        penalty = opp_player_worse_moves - player_worse_moves


    own_moves = len(player_legal_moves)
    opp_moves = len(opp_player_legal_moves)
    return float(own_moves - opp_moves + reward + penalty)


def improved_score_squared(game, player):
    """ Based on the "Improved" evaluation function discussed in lecture that
    outputs a score equal to the difference in the number of moves available to the
    two players. But we calculate the difference of the square value of the #my_moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves_squared = len(game.get_legal_moves(player)) ** 2
    opp_moves_squared = len(game.get_legal_moves(game.get_opponent(player))) ** 2

    return float(own_moves_squared - opp_moves_squared)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # try the improved_score() from the sample_players code base
    return negative_impact_score(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # Check the open book again the legal_moves and keep one handy
        # for now we'll just pick a random one from the available moves.
        best_move = pick_good_move(legal_moves)
        if game.move_count <= 4:
            return best_move

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            curr_best_move = best_move
            curr_best_move_stable_count = 0
            depth = 0;
            if self.iterative:
                while True:
                    depth +=1
                    _, curr_best_move = getattr(self, self.method)(game, depth)
                    if curr_best_move == best_move:
                        curr_best_move_stable_count+=1
                    else:
                        curr_best_move_stable_count = 0
                        best_move = curr_best_move
                    if curr_best_move_stable_count > 5:
                        break
            else: # no ID
                _, best_move = getattr(self, self.method)(game, self.search_depth)
            pass

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        # print("BestMove: ", best_move)
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            # print("TIMEOUT in minimax()")
            raise Timeout()

        if game.is_winner(self):
            return float('+inf'), game.get_legal_moves()

        legal_moves = game.get_legal_moves()

        # check if we reached the required depth or no more moves (terminal node)
        if (depth == 0) or (not legal_moves):
            game_score = self.score(game, self)
            return (game_score, (-1, -1))

        # iterate over every potential game and calculate the best score for the branch
        good_pick = ()
        if maximizing_player:
            good_pick = (float("-inf"), (-1, -1))
            for move in legal_moves:
                game_branch = game.forecast_move(move)
                branch_score, _ = self.minimax(game_branch, depth -1, (not maximizing_player))
                if branch_score > good_pick[0]:
                    good_pick = (branch_score, move)
        else:
            good_pick = (float("+inf"), (-1, -1))
            for move in legal_moves:
                game_branch = game.forecast_move(move)
                branch_score, _ = self.minimax(game_branch, depth -1, (not maximizing_player))
                if branch_score < good_pick[0]:
                    good_pick = (branch_score, move)

        return good_pick


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if game.is_winner(self):
            return float('+inf'), game.get_legal_moves()

        legal_moves = game.get_legal_moves()

        # check if we reached the required depth or no more moves (terminal node)
        if (depth == 0) or (not legal_moves):
            game_score = self.score(game, self)
            return (game_score, (-1, -1))

        good_choice = ()
        if maximizing_player:
            best_score = alpha
            good_choice = (alpha, (-1, -1))
            for move in legal_moves:
                game_branch = game.forecast_move(move)
                branch_score, _ = self.alphabeta(game_branch, depth -1, alpha, beta, (not maximizing_player))
                if branch_score > best_score:
                    good_choice = (branch_score, move)
                    best_score = branch_score
                if best_score >= beta:
                    break
                alpha = max(alpha, best_score)

        else:
            best_score = beta
            good_choice = (beta, (-1, -1))
            for move in legal_moves:
                game_branch = game.forecast_move(move)
                branch_score, _ = self.alphabeta(game_branch, depth -1, alpha, beta, (not maximizing_player))
                if branch_score < best_score:
                    good_choice = (branch_score, move)
                    best_score = branch_score
                if best_score <= alpha:
                    good_choice = (best_score, move)
                    break
                beta = min(beta, best_score)

        return good_choice;