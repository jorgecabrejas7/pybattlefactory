import sys
import os

# Ensure pybattle is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

import pybattle
print(f"DEBUG: pybattle contents: {dir(pybattle)}")
from pybattle import factory_hrl_env

def verify_exclusion():
    print("Verifying bindings for exclusion...")
    env = factory_hrl_env.FactoryHRL_Env(challenge_num=0, is_open_level=True, seed=42)
    obs, info = env.reset()
    
    # We are in RENTAL phase. Pick first 3 mons.
    action = 0 # Combination 0
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Now in BATTLE phase. 
    # Check teams
    player_species = [p.species for p in env.player_team]
    opponent_species = [p.species for p in env.opponent_team]
    
    print(f"Player Team: {player_species}")
    print(f"Opponent Team: {opponent_species}")
    
    # Check for intersection
    intersection = set(player_species).intersection(set(opponent_species))
    if intersection:
        print(f"FAILED: Intersection found: {intersection}")
        sys.exit(1)
    else:
        print("Success: No intersection on first try.")
        
    # Rigorous test: Try many times to ensure logic works (and not just luck)
    # We need to force regeneration of opponent team with collision potential.
    # Player team is random.
    collisions = 0
    trials = 100
    
    print(f"Running {trials} trials...")
    for i in range(trials):
        env.reset(seed=i)
        # Pick 3 rentals
        env.step(0) 
        p_set = set(p.species for p in env.player_team)
        o_set = set(p.species for p in env.opponent_team)
        if not p_set.isdisjoint(o_set):
            collisions += 1
            print(f"Collision in trial {i}: {p_set} vs {o_set}")
            
    if collisions == 0:
        print("PASSED: 0 collisions in 100 trials.")
    else:
        print(f"FAILED: {collisions} collisions in 100 trials.")
        sys.exit(1)

if __name__ == "__main__":
    verify_exclusion()
