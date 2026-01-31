import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys

# Ensure pybattle can be imported
BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build')
if BUILD_DIR not in sys.path:
    sys.path.append(BUILD_DIR)

# LD_PRELOAD fix for libstdc++ in some environments
# Note: This is a hack, usually you'd want to fix the environment or link statically
# os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

try:
    import pybattle
except ImportError:
    # Try direct relative path
    sys.path.append("./simulator/build")
    import pybattle

class PokemonEnv(gym.Env):
    """
    A Gymnasium environment for Pokemon Battle Factory.
    """
    def __init__(self, challenge_num=0, is_open_level=True, seed=None):
        super().__init__()
        self.engine = pybattle.BattleEngine()
        self._seed_val = seed if seed is not None else 0
        self.factory = pybattle.FactoryGenerator(self._seed_val)
        self.challenge_num = challenge_num
        self.is_open_level = is_open_level

        # Action space: 0-3: Move, 4-8: Switch, 9: Struggle
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # 0-1: HP ratios
        # 2-3: Species IDs
        # 4-17: Stat stages (7 per side)
        # 18-21: Move IDs (player active)
        # 22-25: PP ratios (player active)
        # 26-27: Status (player, opponent)
        # 28-29: Remaining party count
        self.observation_space = spaces.Box(
            low=-1.0, high=1000.0, shape=(30,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_val = seed
            self.factory = pybattle.FactoryGenerator(seed)
        
        # Generate teams
        player_pool = self.factory.generate_player_team(self.challenge_num, self.is_open_level)
        player_team = player_pool[:3]
        opp_team = self.factory.generate_opponent_team(self.challenge_num, 1, self.is_open_level)
        
        self.engine.reset(self._seed_val)
        self.engine.set_player_team(player_team)
        self.engine.set_opponent_team(opp_team)
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Validate action
        legal_actions = self.engine.get_legal_actions()
        legal_types = [int(a.type) for a in legal_actions]
        
        # If action is not legal, pick a random legal action or first legal
        if action not in legal_types:
            action = legal_types[0]
            
        pkmn_action = pybattle.Action(pybattle.ActionType(action))
        
        # Step the engine
        result = self.engine.step(pkmn_action)
        
        obs = self._get_obs()
        reward = result.reward
        terminated = result.done
        truncated = False
        info = {"winner": result.winner}
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        state = self.engine.get_state()
        p0 = state.get_active_pokemon(0)
        p1 = state.get_active_pokemon(1)
        a0 = state.get_active(0)
        a1 = state.get_active(1)
        
        obs = np.zeros(30, dtype=np.float32)
        
        # HP Ratios
        obs[0] = p0.current_hp / max(1, p0.max_hp)
        obs[1] = p1.current_hp / max(1, p1.max_hp)
        
        # Species
        obs[2] = float(p0.species) / 412.0 # Normalized
        obs[3] = float(p1.species) / 412.0
        
        # Stat stages
        s0 = a0.stat_stages
        s1 = a1.stat_stages
        for i in range(7):
            obs[4+i] = s0[i] / 6.0 # -1 to 1
            obs[11+i] = s1[i] / 6.0
            
        # Moves
        p0_moves = p0.moves
        p0_pp = p0.pp
        for i in range(min(4, len(p0_moves))):
            obs[18+i] = float(p0_moves[i]) / 355.0
            # PP ratio
            obs[22+i] = p0_pp[i] / 40.0
            
        # Status (Placeholder for now, could use p0.status if exposed)
        obs[26] = 0.0
        obs[27] = 0.0
        
        # Party count
        obs[28] = float(state.get_player_party_count()) / 3.0
        obs[29] = float(state.get_opponent_party_count()) / 3.0
        
        return obs
