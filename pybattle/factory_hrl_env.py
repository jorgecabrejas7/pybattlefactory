import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
import itertools
from enum import IntEnum

# Ensure pybattle can be imported
# The shared library is usually in simulator/build/
BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build')
if BUILD_DIR not in sys.path:
    sys.path.append(BUILD_DIR)

try:
    import pybattle
except ImportError:
    # Try direct relative path if called from project root
    sys.path.append("./simulator/build")
    import pybattle

class FactoryPhase(IntEnum):
    RENTAL = 0
    BATTLE = 1
    SWAP = 2

class FactoryHRL_Env(gym.Env):
    """
    A specialized Gymnasium environment for Pokemon Battle Factory (Emerald).
    Manages the full lifecycle of a challenge (7 battles).
    """
    def __init__(self, challenge_num=0, is_open_level=True, seed=None):
        super().__init__()
        self.engine = pybattle.BattleEngine()
        self._seed_val = seed if seed is not None else 0
        self.factory = pybattle.FactoryGenerator(self._seed_val)
        
        self.challenge_num = challenge_num
        self.is_open_level = is_open_level
        self.current_battle = 0
        self.phase = FactoryPhase.RENTAL
        
        # State variables
        self.player_team = []
        self.opponent_team = []
        self.rental_pool = []
        self.opponent_pool = [] # The pool the opponent mon will come from for swapping
        
        # Rental combinations (20)
        self.rental_combos = list(itertools.combinations(range(6), 3))
        
        # Action space: Discrete(20)
        #   RENTAL: 0-19 (combinations)
        #   BATTLE: 0-9 (moves/switches)
        #   SWAP: 0 (keep), 1-9 (swap P[i//3] for O[i%3])
        self.action_space = spaces.Discrete(20)

        # Observation space (Unified)
        # [0]: Phase (0, 1, 2)
        # [1]: Battle Number (0-6)
        # ... rest similar to pkmn_env.py but expanded for pools
        self.observation_space = spaces.Box(
            low=-1.0, high=1000.0, shape=(100,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_val = seed
        
        # New seed for the challenge
        self.factory = pybattle.FactoryGenerator(self._seed_val)
        self.current_battle = 0
        self.phase = FactoryPhase.RENTAL
        
        # Generate initial rental pool
        self.rental_pool = self.factory.generate_player_team(self.challenge_num, self.is_open_level)
        self.player_team = []
        self.opponent_team = []
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self.phase == FactoryPhase.RENTAL:
            # Handle RENTAL phase
            if action >= 20: action = 0 # Safety
            chosen_indices = self.rental_combos[action]
            self.player_team = [self.rental_pool[i] for i in chosen_indices]
            
            # Start first battle
            self._start_battle()
            
        elif self.phase == FactoryPhase.BATTLE:
            # Handle BATTLE phase
            legal_actions = self.engine.get_legal_actions()
            legal_types = [int(a.type) for a in legal_actions]
            
            # Action masking safety
            if action not in legal_types:
                action = legal_types[0]
                
            pkmn_action = pybattle.Action(pybattle.ActionType(action))
            state = self.engine.get_state()
            
            # Prevent infinite loops in random verification
            if state.turn_number > 200:
                truncated = True
                result = self.engine.step(pkmn_action)
            else:
                result = self.engine.step(pkmn_action)
            
            if result.done:
                if result.winner == 0: # Player won
                    reward = 1.0 # Base win reward
                    if self.current_battle == 6: # Challenge complete
                        terminated = True
                    else:
                        self.phase = FactoryPhase.SWAP
                else: # Player lost
                    reward = -1.0
                    terminated = True
            
        elif self.phase == FactoryPhase.SWAP:
            # Handle SWAP phase
            # Action 0: Keep team
            # Action 1-9: Swap P[(action-1)//3] for O[(action-1)%3]
            if action > 0 and action <= 9:
                p_idx = (action - 1) // 3
                o_idx = (action - 1) % 3
                # Swap
                self.player_team[p_idx] = self.opponent_team[o_idx]
            
            self.current_battle += 1
            self._start_battle()

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _start_battle(self):
        # Generate opponent team
        self.opponent_team = self.factory.generate_opponent_team(
            self.challenge_num, 
            self.current_battle + 1, 
            self.is_open_level
        )
        
        # Reset engine
        self.engine.reset(self._seed_val + self.current_battle + 1)
        self.engine.set_player_team(self.player_team)
        self.engine.set_opponent_team(self.opponent_team)
        
        self.phase = FactoryPhase.BATTLE

    def _get_obs(self):
        obs = np.zeros(100, dtype=np.float32)
        obs[0] = float(self.phase)
        obs[1] = float(self.current_battle)
        
        # Action Mask (indices 0-19)
        mask = np.zeros(20, dtype=np.float32)
        
        if self.phase == FactoryPhase.RENTAL:
            # All 20 rental combos are valid
            mask[:] = 1.0
            
            # Add rental pool to obs (indices 2-61: 6 mons * 10 features)
            for i, p in enumerate(self.rental_pool):
                self._write_mon_obs(obs, 2 + i*10, p)
                
        elif self.phase == FactoryPhase.BATTLE:
            # Mask based on legal actions from engine
            legal_actions = self.engine.get_legal_actions()
            for a in legal_actions:
                idx = int(a.type)
                if idx < 20: mask[idx] = 1.0
            
            # Active battle obs (indices 2-61 already used by pools, let's use 62+)
            state = self.engine.get_state()
            p0 = state.get_active_pokemon(0)
            p1 = state.get_active_pokemon(1)
            a0 = state.get_active(0)
            a1 = state.get_active(1)
            
            self._write_mon_obs(obs, 62, p0, a0)
            self._write_mon_obs(obs, 72, p1, a1)
            
            # Team sizes/status
            obs[82] = float(state.get_player_party_count())
            obs[83] = float(state.get_opponent_party_count())
            
        elif self.phase == FactoryPhase.SWAP:
            # Action 0 (Keep) always valid
            mask[0] = 1.0
            # Actions 1-9 (Swaps) valid if within team sizes (3x3 = 9)
            mask[1:10] = 1.0
            
            # Show current team vs opponent team
            for i, p in enumerate(self.player_team):
                self._write_mon_obs(obs, 2 + i*10, p)
            for i, p in enumerate(self.opponent_team):
                self._write_mon_obs(obs, 32 + i*10, p)

        # Append mask to obs for convenience (optional, but requested indices)
        # Actually let's put it at the end
        obs[80:100] = mask
        
        return obs

    def _write_mon_obs(self, obs, start_idx, p, a=None):
        # 10 features per mon
        obs[start_idx] = float(p.species) / 412.0
        obs[start_idx+1] = float(p.current_hp) / max(1.0, float(p.max_hp))
        obs[start_idx+2] = float(p.level) / 100.0
        # Moves (4)
        for i in range(4):
            obs[start_idx+3+i] = float(p.moves[i]) / 355.0
        # Stat stages if active
        if a:
            # Just use Speed stage as a representative or first one?
            # Or average? Let's just use first 3 stat stages
            stages = a.stat_stages
            obs[start_idx+7] = stages[0] / 6.0
            obs[start_idx+8] = stages[1] / 6.0
            obs[start_idx+9] = stages[2] / 6.0
        else:
            obs[start_idx+7:start_idx+10] = 0.0
