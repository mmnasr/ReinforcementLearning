import copy

class Grid:
    def __init__(self, n_rows=3, n_cols=3, start_idx=(0,0)):
        self.rows = n_rows
        self.cols = n_cols
        # i: row idx
        self._i = start_idx[0]
        # j: col idx
        self._j = start_idx[1]
        self._all_states = {}
        self._rewards = {}
        self._actions = {}

    def print_info(self):
        print("Grid world information:")
        print("  # of rows: {}".format(self.rows))
        print("  # of cols: {}".format(self.cols))
        print("  Current index (i,j) = ({},{})".format(self._i, self._j))
        print("  Actions:")
        for a in self._actions:
            print("      {}: {}".format(a, self._actions[a]))
        print("  Rewards:")
        for r in self._rewards:
            print("      {}: {}".format(r, self._rewards[r]))

    def set_rewards(self, rewards):
        """
        Set rewards. Rewards is a dictionary {key: val}
            - Key: state (i,j) tuples
            - Value: reward_value (a scalar)
        """
        self._rewards = copy.deepcopy(rewards)

    def get_reward(self, st):
        """
        Return the reward value for a given state.
        st: a tuple (i,j).
        if st not found in rewards dictionary, return 0.
        """
        if st in self._rewards:
            return self._rewards[st]
        # print("Warning. Could not find state {} in rewards dictionary. Returning 0 reward.")
        # set default reward as zero
        return 0

    def set_actions(self, actions):
        """
        Set actions. Actions is a dictionary {key: val}
            - Key: state (i,j) tuples
            - Value: A (a list of possible actions from current cell at (i,j))
        """
        self._actions = copy.deepcopy(actions)

    def get_actions(self, state):
        if state in self._actions:
            return self._actions[state]
        return None

    def set_state(self, st):
        """
        Set a new state. For this environment, it's the the index of the new cell at (i,j).
        st is a tuple
        """
        self._i = st[0]
        self._j = st[1]

    def get_current_state(self):
        """
        Return current state as an index tuple: (i,j).
        """
        return (self._i, self._j)

    def get_all_states(self):
        return set(self._actions.keys()) | set(self._rewards.keys())

    def _update_state_using_action(self, action, mode='default'):
        """
        Updates the index (i,j) given action.
        Accepted values for 'action':
            'U': Up
            'D': Down
            'L': Left
            'R': Right
        Accepted values for mode:
            'default': moves forward via action.
            'undo': moves the piece backwards (to reset the grid back to the original place)
        """
        # mapping dictionay between actions and increments from (i,j) to new cell.
        action_increment = {
            'U': (+1,0),
            'D': (-1,0),
            'R': (0,+1),
            'L': (0,-1)
        }
        action_inc = action_increment[action.upper()]
        # Default mode: Move the piece forward
        if mode.lower() == 'default':
            self._i += action_inc[0]
            self._j += action_inc[1]
        # Undo move mode. Move the piece backwards to the original place.
        elif mode.lower() == 'undo':
            self._i -= action_inc[0]
            self._j -= action_inc[1]
        else:
            raise ValueError('Unexpected value for "mode": {}. Use "default" or "undo".'.format(mode))

    def move(self, action):
        """
        Update the state (i,j) based on the given action.
        Return:
            reward value for the updated state (after taking the action.)
        """
        # return (i,j) tuple as current state
        current_state = self.get_current_state()
        if action not in self._actions[current_state]:
            raise ValueError('Cannot move to new cell. Action {} not in possible actions for cell {}.'.format(action, current_state))
        else:
            # move to the next cell based on current action.
            self._update_state_using_action(action, mode='default')

        # return the reward value for the new state
        return self.get_reward((self._i,self._j))

    def undo_move(self, action):
        """
        Undo move based on the action.
        """
        self._update_state_using_action(action, mode='undo')
        # if current state no in all states, then there is an error.
        assert(self.get_current_state() in self.get_all_states())

    def is_terminal(self, st):
        """
        If current state not in action keys, it's a terminal state.
        TODO: There should be a better way to define this.
        """
        if st not in self._actions.keys():
            return True
        return False

    def game_over(self):
        """
        If current state not in action keys (states), game over.
        We have reached a terminal state.
        """
        if self.get_current_state() not in self._actions.keys():
            return True
        return False

# Create standard grid.
def create_standard_grid(n_rows=3, n_cols=4, start_idx=(0,0),\
                         banned_cells=[(1,1)],\
                         terminal_cells_and_rewards={(2,3):1, (1,3):-1},\
                         step_negative_reward=None):

    def is_setup_correct():
        """
        Do sanity checks on the given grid setup.
        """
        if not terminal_cells_and_rewards:
            print('Error. reward_cells dictionary cannot be empty. Set at least two cells with positive and negative rewards.')
            return False

        if any(_out_of_bounds(cell) for cell in terminal_cells_and_rewards.keys()):
            print('Error. Reward cell dictionary includes out-of-bounds cell.')
            return False

        if banned_cells and any(_out_of_bounds(cell) for cell in banned_cells):
            print('Error. Banned cell includes out-of-bounds cell.')
            return False

        return True

    def _out_of_bounds(cell_idx):
        i = cell_idx[0]
        j = cell_idx[1]
        if i >= n_rows or i < 0 or j >= n_cols or j < 0:
            return True
        return False

    def _is_banned_cell(cell_idx):
        if banned_cells and cell_idx in banned_cells:
            return True
        return False

    def _is_reward_cell(cell_idx):
        if cell_idx in terminal_cells_and_rewards.keys():
            return True
        return False

    def _generate_possible_actions():
        """
        Go through entire grid. Create a dictionay of
        cells (as keys) and a list of feasible actions.
        Action dictionary should NOT include terminal cells/states.
        Also, it will not include the 'banned cells'. These are 'dead cells' which user is not allowed to enter.
        Returns:
        A dictionary in the form of:
            {(i,j): ['U', 'D', ...]}
        """
        action_increment = {
            'U': (+1,0),
            'D': (-1,0),
            'R': (0,+1),
            'L': (0,-1)
        }
        actions = {}
        for i in range(n_rows):
            for j in range(n_cols):
                current_cell = (i,j)
                # If current cell is a reward_cell (terminal state) or a banned cell (not allowed cell),
                # don't add it to action dictionay.
                if _is_reward_cell(current_cell) or _is_banned_cell(current_cell):
                    continue
                feasible_action_list = []
                for act in action_increment:
                    increment = action_increment[act]
                    next_cell = (i+increment[0], j+increment[1])
                    if not _is_banned_cell(next_cell) and not _out_of_bounds(next_cell):
                        feasible_action_list.append(act)
                actions[current_cell] = feasible_action_list
        return actions

    if not is_setup_correct():
        print('Error setting up the grid.')
        return None

    # create the grid (environment)
    g = Grid(n_rows=n_rows, n_cols=n_cols, start_idx=start_idx)

    # generate a dictionary of possible action for each cell
    actions = {}
    actions = _generate_possible_actions()
    g.set_actions(actions)

    rewards = {}
    rewards = copy.deepcopy(terminal_cells_and_rewards)
    # if has negative reward for any basic move (that won't end up at a terminal state)
    # add the constant reward to the reward dictionary.
    if step_negative_reward:
        augment_rewards_with_constants = {}
        augment_rewards_with_constants = {cell:step_negative_reward for cell in actions.keys()}
        rewards.update(augment_rewards_with_constants)
    g.set_rewards(rewards)

    return g
