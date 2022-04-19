import numpy as np

from .tiles3 import tiles, IHT


class Representation(object):
    """
    Dummy representation interface
    """
    def __init__(self, cfg):
        self.num_actions = cfg.action_dim
        self.num_obs = cfg.state_dim

    def get_num_features(self):
        return self.acts * self.obs

    def get_representation(self, obs, action):
        """
        Examples:
        obs: [2, 3] with action 1 (out of 3) returns:
            [0, 0, 2, 3, 0, 0]

        obs: [2, 3] with action 2 returns:
            [0, 0, 0, 0, 2, 3]
        """
        rep = np.zeros(self.num_obs * self.num_actions)
        rep[action * self.num_obs: (action + 1) * self.num_obs] = obs
        return rep


class TileCoder(Representation):
    """
    Tile Coding representation
    Config should have three attributes:
        mem_size: maximum size of the hash table memory
        tiles: number of tiles to separate each dimension into
        tilings: number of tilings of each dimension
        tile_combinations: if combinations of the state components are
                           to be tiled separately
    """
    def __init__(self, cfg):
        super(TileCoder, self).__init__(cfg)
        self.tiles = cfg.tiles
        self.tilings = cfg.num_tilings
        self.tile_separate = cfg.tile_separate
        self.combinations = cfg.combinations if self.tile_separate else []
        self.mem_size = cfg.tiles_memsize

        if self.tile_separate:
            self.com_mem = self.mem_size // len(self.combinations)
            self.iht = []
            for k in range(len(self.combinations)):
                self.iht.append(IHT(self.com_mem))
        else:
            self.iht = IHT(self.mem_size)

    def get_num_features(self):
        return self.mem_size

    def get_representation(self, state, action=None):
        # Assumption: state components are -1, 1 normalized
        ones = []
        if self.tile_separate:
            for k, p in enumerate(self.combinations):
                ints = [] if action is None else [action]
                tc = tiles(self.iht[k], self.tilings, float(self.tiles) * np.array([state[m] for m in p]), ints=ints)
                tc = [x + (self.com_mem * k) for x in tc]
                ones.extend(list(tc))
        else:
            ints = [] if action is None else [action]
            # tc = tiles(64, self.tilings, self.scale * state, ints=ints)
            tc = tiles(self.mem_size, self.tilings, float(self.tiles) * state, ints=ints)
            ones.extend(list(tc))
        rep = np.zeros(self.mem_size)
        rep[ones] = 1
        return rep
