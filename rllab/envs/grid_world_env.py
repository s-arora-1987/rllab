import numpy as np
from .base import Env
from rllab.spaces import Discrete
from rllab.envs.base import Step
from rllab.core.serializable import Serializable

MAPS = {
    "chain": [
        "GFFFFFFFFFFFFFSFFFFFFFFFFFFFG"
    ],
    "4x4_safe": [
        "SFFF",
        "FWFW",
        "FFFW",
        "WFFG"
    ],
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "Patrol": [
        "WFFFFFWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFWWWWWWW",
        "WFFFFFWWW",
    ],
}


class GridWorldEnv(Env, Serializable):
    """
    'S' : starting point
    'F' or '.': free space
    'W' or 'x': wall
    'H' or 'o': hole (terminates episode)
    'G' : goal



    """

    def __init__(self, desc='4x4'):
        Serializable.quick_init(self, locals())
        self.desc_str = desc
        if isinstance(desc, str):
            desc = MAPS[desc]
        desc = np.array(list(map(list, desc)))
        desc[desc == '.'] = 'F'
        desc[desc == 'o'] = 'H'
        desc[desc == 'x'] = 'W'
        self.desc = desc
        self.n_row, self.n_col = desc.shape
        self.valid_states = []
        self.state_enum = {}
        i=0
        if self.desc_str == "Patrol":
            for x in range(0,self.n_row):
                for y in range(0,self.n_col):
                    for th in range(0,4):
                        if self.desc[x, y] == 'F':
                            state=(x*self.n_col+y)*4+th
                            self.valid_states.append(state)
                            self.state_enum[i]=state
                            i=i+1
        print(self.state_enum)
        print(list(self.state_enum.keys()))

        if self.desc_str == "Patrol":
            start_x, start_y, start_th = 0, 1, 0
            start_state = (start_x * self.n_col + start_y)*4+ start_th
            self.start_state = list(self.state_enum.keys())[list(self.state_enum.values()).index(start_state)]
        else:
            (start_x,), (start_y,) = np.nonzero(desc == 'S')
            self.start_state = start_x * self.n_col + start_y

        self.state = None
        self.domain_fig = None


    def reset(self):
        if self.desc_str == "Patrol":
            self.state = np.random.choice(list(self.state_enum.keys()))
        else:
            self.state = self.start_state
        return self.state

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction. This is a helper method for debugging and testing
        purposes.
        :return: the action index corresponding to the given direction
        """
        return dict(
            left=0,
            down=1,
            right=2,
            up=3
        )[d]

    def step(self, action):
        """
        action map:
        0: left
        1: down
        2: right
        3: up
        :param action: should be a one-hot vector encoding the action
        :return:
        action map for Patrol:
        0: turn-left
        1: forward
        2: turn-right
        3: stop

        """
        possible_next_states = self.get_possible_next_states(self.state, action)

        probs = [x[1] for x in possible_next_states]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_state = possible_next_states[next_state_idx][0]  # an index for state_enum in Patrol

        if self.desc_str == "Patrol":
            next_state = self.state_enum[next_state]
            next_x = (next_state // 4) // self.n_col
            next_y = (next_state // 4) % self.n_col
        else:
            next_x = next_state // self.n_col
            next_y = next_state % self.n_col

        next_state_type = self.desc[next_x, next_y]
        if next_state_type == 'H':
            done = True
            reward = 0
        elif next_state_type in ['F', 'S']:
            done = False
            if self.desc_str == "Patrol" and next_state_type == 'F':
                x = (self.state // 4) // self.n_col
                y = (self.state // 4) % self.n_col
                if x != next_x or y != next_y:
                    if next_y == 3: # or next_y == 5 or next_y == 6:
                        reward = 0.57+0.35  # 0.43
                    else:
                        reward = 0.57
                else:
                    reward = 0
        elif next_state_type == 'G':
            done = True
            reward = 1
        else:
            raise NotImplementedError

        if self.desc_str == "Patrol":
            # switch back from value to key
            self.state = list(self.state_enum.keys())[list(self.state_enum.values()).index(next_state)]
        else:
            self.state = next_state

        return Step(observation=self.state, reward=reward, done=done)

    def get_possible_next_states(self, in_state, action):
        """
        Given the state and action, return a list of possible next states and their probabilities. Only next states
        with nonzero probabilities will be returned
        :param state: start state
        :param action: action
        :return: a list of pairs (s', p(s'|s,a))
        """
        # assert self.observation_space.contains(state)
        # assert self.action_space.contains(action)


        if self.desc_str == "Patrol":

            state = self.state_enum[in_state] # map from enumeration to row-major state value
            x = (state // 4) // self.n_col
            y = (state // 4) % self.n_col
            th = state % 4
            if action == 0:
                next_th = th+1
                if next_th > 3:
                    next_th = 0
                next_x = x
                next_y = y
                # next_state = (state // 4)*4 + next_th
            elif action == 1:
                if (th == 0):
                    next_y = y + 1
                    next_x = x
                    # next_state = (x* self.n_col +  y + 1)*4 + th
                if (th == 1):
                    next_x =next_state = x - 1
                    next_y = y
                    # next_state = ((x - 1) * self.n_col + y)*4 + th
                    # s[0] -= 1
                if (th == 2):
                    next_y = y - 1
                    next_x = x
                    # next_state = (x * self.n_col + y - 1)*4 + th
                    # s[1] -= 1
                if (th == 3):
                    next_x = x + 1
                    next_y = y
                    # next_state = ((x + 1) * self.n_col + y)*4 + th
                    # s[0] += 1
                next_th = th
            elif action == 2:
                next_th = th - 1
                if next_th < 0:
                    next_th = 3
                next_x = x
                next_y = y
                # next_state = (state // 4)*4 + next_th
            else:
                next_x = x
                next_y = y
                next_th = th

            if next_x < 0 or next_x > self.n_row - 1:
                next_x = x
            if next_y < 0 or next_y > self.n_col - 1:
                next_y = y

            # coords = np.array([next_x, next_y])
            # next_coords = np.clip(
            #     coords,
            #     [0, 0],
            #     [self.n_row - 1, self.n_col - 1]
            # )
            next_state = (next_x * self.n_col + next_y)*4 + next_th
            state_type = self.desc[x, y]

            if next_x >= 16  or next_y >= 9:
                print("")
                # pass
            next_state_type = self.desc[next_x, next_y]


            if next_state_type == 'W' or state_type == 'H' or state_type == 'G':
                return [(in_state, 1.)]
            else:
                next_in_state =list(self.state_enum.keys())[list(self.state_enum.values()).index(next_state)]
                return [(next_in_state, 0.925), (in_state, 0.075)]

        else:
            x = in_state // self.n_col
            y = in_state % self.n_col
            coords = np.array([x, y])

            increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
            next_coords = np.clip(
                coords + increments[action],
                [0, 0],
                [self.n_row - 1, self.n_col - 1]
            )
            next_in_state = next_coords[0] * self.n_col + next_coords[1]
            in_state_type = self.desc[x, y]
            next_in_state_type = self.desc[next_coords[0], next_coords[1]]

            if next_in_state_type == 'W' or in_state_type == 'H' or in_state_type == 'G':
                return [(in_state, 1.)]
            else:
                return [(next_in_state, 1.)]

    @property
    def action_space(self):
        return Discrete(4)

    @property
    def observation_space(self):
        if self.desc_str == "Patrol":
            return Discrete(len(self.state_enum))
        else:
            return Discrete(self.n_row * self.n_col)

