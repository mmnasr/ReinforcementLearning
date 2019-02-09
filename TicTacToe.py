# Tic Tac Toe game
# Using Reinforcement Learning and Epsilon-Greedy algoirthm
# Inspired by LazyProgrammer & Udemy's online course
# By: Mohamad Nasr-Azadani. mmnasr@gmail.com


from __future__ import print_function
import numpy as np
import sys

# Board size
LENGTH = 3
# Human goes first
HUMAN_FIRST = False

class Environment():
    def __init__(self):
        self.board = np.zeros((LENGTH,LENGTH), dtype=int)
        self.X = -1
        self.O = +1
        self.Empty = 0
        self._signs = {1: 'O', -1: 'X', 0: ' '}
        self.winner = None
        self.ended = False
        self.num_states = 3**(LENGTH*LENGTH)
    
    def is_empty(self, i, j):
        return self.board[i,j] == self.Empty
    
    def game_over(self, force_check=False):
        def _win_by_row(which_player):
            """
            Check if player won by completing a row
            """
            win_sum_value = which_player*LENGTH
            for i in range(LENGTH):
                if self.board[i].sum() == win_sum_value:
                    return True
            return False
        
        def _win_by_column(which_player):
            """
            Check if player won by completing a column
            """
            win_sum_value = which_player*LENGTH
            for j in range(LENGTH):
                if self.board[:, j].sum() == win_sum_value:
                    return True
            return False
        
        def _win_by_diagonal(which_player):
            """
            Check if player won by completing a diagonal
            """
            win_sum_value = which_player*LENGTH
            sum_diag1 = 0
            sum_diag2 = 0
            for i in range(LENGTH):
                # sum of main diagonal
                sum_diag1 += self.board[i,i] # main diagonal
                # sum of the other diagonal (top-right to botto left)
                sum_diag2 += self.board[i,LENGTH-i-1]
            if sum_diag1 == win_sum_value or sum_diag2 == win_sum_value:
                return True
            
            return False
        
        def _is_board_full():
            """
            Check if (LENGTH*LENGTH) board is full.
            """
            if np.all( (self.board == self.Empty) == False):
                return True
            return False

        # If game is already ended and we are not forcing for another check, 
        # then just return the ended flag
        if not force_check and self.ended:
            return self.ended
        
        for p in (self.X,self.O):
            if _win_by_row(which_player=p) or _win_by_column(which_player=p) or _win_by_diagonal(which_player=p):
                self.winner = p
                self.ended = True
                return True
            
        # If we made it here, then there is definitely no winner.
        self.winner = None

        # If board is full, then game over, else, continue the game.
        self.ended = _is_board_full()
        return self.ended
    
    def is_tie(self):
        """
        Returns if the game is tied (ended and no winner)
        """
        return self.ended and self.winner is None
    
    def show_board(self):
        def _get_char(i,j):
            return self._signs[self.board[i,j]]
            
        for i in range(LENGTH):
            print('-'*(4*LENGTH+1))
            s = ['|']
            for j in range(LENGTH):
                char = _get_char(i,j)
                s.append(char)
                s.append('|')
            print(' '.join(s))
        print('-'*(4*LENGTH+1))
    
    def get_state(self):
    # returns the current state, represented as an int
    # from 0...|S|-1, where S = set of all possible states
    # |S| = 3^(BOARD SIZE), since each cell can have 3 possible values - empty, x, o
    # some states are not possible based on rule's game, e.g. all cells are x, 
    # but we ignore that detail. 
    # This is like finding the integer represented by a base-3 number
        def _get_val(i,j):
            """
            Assign a unique value for each cell-state. 
            X = 1, O = 2, Empty = 0
            """
            if self.board[i,j] == self.X:
                return 1
            if self.board[i,j] == self.O:
                return 2
            return 0
        # Go through the board, and create a single number from the values stored.
        # Assign unique values for each case. 
        k, h = 0, 0
        for i in range(LENGTH):
            for j in range(LENGTH):
                v = _get_val(i,j)
                h += (3**k) * v
                k += 1
        return h
    
    def get_winner(self):
        return self.winner
    
    def set_cell(self, idx_tuple, which_player):
        i = idx_tuple[0]
        j = idx_tuple[1]
        if not self.is_empty(i,j):
            raise ValueError('Error. Cannot place {} in cell at (i,j)=({},{}).            Cell already occupied.'.format(which_player, i, j))
        else:
            self.board[i,j] = which_player
            
    def reset_cell(self, idx_tuple):
        i = idx_tuple[0]
        j = idx_tuple[1]
        self.board[i,j] = self.Empty

    def reward(self, which_player):
        # If game is still on, just pass 0 as reward
        if not self.game_over():
            return 0
        # Game is ended. Check if winner is which_player, give a positive reward.
        # If not, a 0 reward
        if self.get_winner() == which_player:
            return 1
        # Question: What happens to draw or if other player wins? 
        # Why no negative reward?
        return 0


# In[4]:


class Agent:
    def __init__(self, which_player=None, eps=0.1, alpha=0.5):
        # epsilon: in Epsilon-Greedy algorithm.
        # Choose a small number. The probability of choosing a random action.
        self._eps = eps
        # learning rate
        self._alpha = alpha
        # array tracking state histroy of the game
        self.state_history = []
        if not which_player:
            raise ValueError('which_player cannot be None. Set either "X" or "O" for different players')
        else:
            self._set_player(which_player)
    
    def setVs(self, V=[]):
        self._V = V
    
    def getV_for_state(self, state):
        return self._V[state]

    def _set_player(self, p):
        if p in ['X', 'x']:
            self._player = -1
        elif p in ['O', 'o']:
            self._player = +1
        else:
            raise ValueError('Error. Unexpected player symbol {}. Use "X" or "O" '.format(p))
        
    def get_player(self):
        return self._player
    
    def get_player_symbol(self):
        return 'X' if self._player == -1 else 'O'

    def reset_history(self):
        self.state_history = []

    def take_action(self, env, show_greedy_values=False):
        def _explore(env):
            """
            Explore step. Pick an empty cell on the board randomly.
            """
            def __possible_cells():
                empty_cells = []
                for i in range(LENGTH):
                    for j in range(LENGTH):
                        if env.is_empty(i,j):
                            empty_cells.append((i,j))
                return empty_cells

            # First, collect all possible cells (non-empty) on the board
            empty_cells = __possible_cells()
            # Now pick one randomly
            idx = np.random.choice(len(empty_cells))
            # Next, store the randomly chosen empty cell index (i,j) on the board
            return empty_cells[idx]
        
        def _exploit(env):
            """
            Exploit step. Go through all possible actions (next empty cell),
            with the possible next move, find the state and value for new state. 
            Return the best next cell with the most value.
            """
            def __print_values():
                print("Taking a greedy action:")
                for i in range(LENGTH):
                    print('-'*(7*LENGTH+1))
                    s = ['|']
                    for j in range(LENGTH):
                        if env.is_empty(i,j):
                            v = format(pos2value[(i,j)], '.2f')
                            s.append(v)
                        elif env.board[i,j] == env.X:
                            s.append('  X ')
                        elif env.board[i,j] == env.O:
                            s.append('  O ')
                        s.append('|')
                    print(' '.join(s))
                print('-'*(7*LENGTH+1))
                
            V_best = -10
            next_cell = None
            # for showing values only
            pos2value = {}
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i,j):
                        env.set_cell((i,j), which_player=self.get_player())
                        Poss_state = env.get_state()
                        V_poss = self.getV_for_state(Poss_state)
                        pos2value[(i,j)] = V_poss
                        if V_poss > V_best:
                            V_best = V_poss
                            State_best = Poss_state
                            next_cell = (i,j)
                        # now reset the cell again
                        env.reset_cell((i,j))
            if show_greedy_values:
                __print_values()
                    
            return next_cell
        
        # Choose an action using epsilon-greedy algorithm
        r = np.random.rand()
        # cell index: (i,j)
        next_cell = None
        if r < self._eps:
            next_cell = _explore(env)
        else:
            next_cell = _exploit(env)
            
        which_player = self.get_player()
        env.set_cell(next_cell, which_player)
    
    def update_state_history(self, st):
        """
        Append the latest state to state_history list
        """
        self.state_history.append(st)
    
    def update(self, env):
        # we want to BACKTRACK over the states, so that:
        # V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
        # where V(next_state) = reward if it's the most current state
        #
        # NOTE: we ONLY do this at the end of an episode (game ended)
        # (not so for all the algorithms we will study)

        # First, get the reward at this (final) state.
        reward = env.reward(which_player=self.get_player())
        target = reward
        # Now, go through all the preview states (from first move to last move)
        # update their values using the final 'reward'.
        for prev_state in reversed(self.state_history):
            V_prev = self.getV_for_state(prev_state)
            value = V_prev + self._alpha*(target - V_prev)
            self._V[prev_state] = value
            target = value
        # Since we have reached to the end of the game, we reset state history. 
        self.reset_history()


# In[5]:


class Human:
    def __init__(self, which_player=None):
        if not which_player:
            raise ValueError('which_player cannot be None. Set either "X" or "O" for different players')
        else:
            self._set_player(which_player)

    def _set_player(self, p):
        if p in ['X', 'x']:
            self._player = -1
        elif p in ['O', 'o']:
            self._player = 1
        else:
            raise ValueError('Error. Unexpected player symbol {}. Use "X" or "O" '.format(p))
        
    def get_player(self):
        return self._player

    def get_player_symbol(self):
        return 'X' if self._player == -1 else 'O'
    
    def take_action(self, env, show_greedy_values=False):
        while True:
            # break if we make a legal move
            move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
            if sys.version_info[0] < 3:
                i, j = move[0], move[1]
            else:
                i, j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty(i, j):
                env.board[i,j] = self.get_player()
                break
            else:
                print("Try again. Cell is already taken.")

    def update(self, env):
        pass

    def update_state_history(self, st):
        pass


# In[6]:


def play_game(p1, p2, env, show_board=False, show_which_player=None, show_greedy_values=False):
    """
    Play game. We assume p1 and p2 are of types Agent().
    For training phase, p1 and p2 should be Agent(). 
    For testing, p2 can be of class type Human(). 
    """
    def _update_current_player(p1, p2, current_player):
        """
        Swap player's turn
        """
        return p2 if current_player == p1 else p1
        
    current_player = None
    while not env.game_over():
        # Alternate between p1 and p2. 
        # p1 starts first
        current_player = _update_current_player(p1, p2, current_player)
        
        # draw the board if needed
        if show_board: 
            if show_which_player==1 and current_player == p1:
                env.show_board()
            if show_which_player==2 and current_player == p2:
                env.show_board()
            
        # current player makes a new move
        current_player.take_action(env, show_greedy_values=show_greedy_values)
        
        # get current state
        state = env.get_state()
        # allow player 1 and 2 to update their state history accordingly
        p1.update_state_history(state)
        p2.update_state_history(state)
    
    # game ended
    if show_board:
        print('*'*25)
        print('Game ended.')
        env.show_board()
        
    p1.update(env)
    p2.update(env)
    
    return env.ended,env.winner


# In[7]:


def get_states_and_winner(env, i=0, j=0):
    # recursive function that will return all
    # possible states (as ints) and who the corresponding winner is for those states (if any)
    # (i, j) refers to the next cell on the board to permute (we need to try -1, 0, 1)
    # impossible games are ignored, i.e. 3x's and 3o's in a row simultaneously
    # since that will never happen in a real game
    results = []
    
    for v in (env.X, env.O, env.Empty):
        env.board[i,j] = v
        if j == LENGTH-1:
        # j goes back to 0, increase i, unless i = LENGTH-1, then we are done
            if i == LENGTH-1:
                state = env.get_state()
                ended = env.game_over(force_check=True)
                winner = env.get_winner()
                results.append((state,winner,ended))
            else: # i
                # increment i
                results += get_states_and_winner(env, i+1, 0)
        else: # j
            # increment j
            results += get_states_and_winner(env, i, j+1)
            
    return results


# In[8]:


def initialize_V(env, state_winner_triples, which_player=None):
    """
    Initialize Value functions from triples (state,winner,ended).
    State is a value representing the board.
  # initialize state values as follows
  # if x wins, V(s) = 1
  # if x loses or draw, V(s) = 0
  # otherwise, V(s) = 0.5
    """
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            v = 1 if winner == which_player else 0
        else:
            v = 0.5
        V[state] = v
    return V


# In[9]:


def train_players(player1, player2, env, training_count=5000):
    # First, get triples of all possible states of the game:
    # (state,ended,winner)
    state_winner_triples = get_states_and_winner(env)

    # Now initialize Value function for player 1
    V1 = initialize_V(env, state_winner_triples, which_player=player1.get_player())
    player1.setVs(V1)

    # Initialize Value function for player 2
    V2 = initialize_V(env, state_winner_triples, which_player=player2.get_player())
    player2.setVs(V2)

    for t in range(training_count):
        ended, winner = play_game(player1, player2, Environment())
        if t % 500 == 0:
            print('Training {} games.'.format(t))
    print('Trainng completed after {} practice rounds.'.format(training_count))

    return player1, player2


if __name__ == '__main__':

    # np.random.seed(0)
    p1 = Agent(which_player='X')
    p2 = Agent(which_player='O')

    env = Environment()
    p1, p2 = train_players(p1, p2, env, training_count=10000)

    if HUMAN_FIRST:
        human = Human(which_player='X')
    else:
        human = Human(which_player='O')
        
    # Play until player types N or n.
    while True:
        print('*'*30)
        print("Game started ***************************.")
        print('*'*30)
        if HUMAN_FIRST:
            ended, winner = play_game(human, p2, Environment(), show_board=True, show_which_player=1, show_greedy_values=True)
        else:
            ended, winner = play_game(p1, human, Environment(), show_board=True, show_which_player=2, show_greedy_values=True)
        print("Game over ******************************.")
        if not winner:
            print("Draw. Play again? [Y/n]")
        else:
            symb = 'X' if winner == -1 else 'O'
            print('Player "{}" won. Play again? [Y/n]'.format(symb))
        answer = input()
        if answer and answer.lower()[0] == 'n':
            break