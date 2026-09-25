#ifndef GEN3_SEARCH_HOST_H
#define GEN3_SEARCH_HOST_H

// Search support for the gen3 battle core (src/gen3/search_host.c): battle outcome, determinization
// of the opponent's hidden information and of the hidden random state of the turn. Usable from C and C++.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One opponent party slot, rebuilt from an explicit, player-knowledge draw (rl/determinize.py).
struct Gen3DetMon {
    int32_t partySlot;      // gEnemyParty index 0-2
    int32_t species;        // <= 0: keep this slot as it is (fainted)
    int32_t moves[4];       // MOVE_NONE for empty slots
    int32_t item;           // held item (ITEM_NONE allowed)
    int32_t ivs[6];         // HP, Atk, Def, Speed, SpAtk, SpDef (0-31)
    int32_t evs[6];         // same order (0-255, total <= 510)
    int32_t nature;         // 0-24
    int32_t abilityBit;     // personality & 1 (only matters when the species has two abilities)
    float hpFraction;       // (0,1]: HP = round(maxHP * f) (min 1); < 0: keep the current HP fraction
};

// What the player has seen of the random-length counters (move attempts / turns elapsed; 0 = none seen).
struct Gen3DetHidden {
    int32_t sleepElapsed[2][3];     // per side (0 player, 1 opponent), party slot 0-2: attempts while asleep
    int32_t confusionElapsed[2];    // per battler: attempts while confused
    int32_t wrapElapsed[2];         // turns wrapped (Wrap, Bind, Fire Spin...)
    int32_t uproarElapsed[2];
    int32_t rampageElapsed[2];      // Outrage / Thrash / Petal Dance lock
};

uint8_t Gen3_BattleOutcome(void);   // gBattleOutcome (0 while the battle runs)

// Rebuild gEnemyParty[slot] for each spec, keeping status, PP spent on shared moves, a consumed item
// and HP (see hpFraction); patches gBattleMons[1] when the slot is the enemy's active battler.
// hiddenSeed >= 0 also resamples, from `hidden` (may be NULL: nothing elapsed), every hidden counter
// that the player cannot see: sleep and confusion (both sides), Disable / Encore timers, wrap /
// uproar / rampage turns, the opponent's Substitute HP, pending Future Sight / Doom Desire damage
// and the opponent's Choice Band lock. Returns 0 on success, -1 on a bad spec (nothing changed),
// and 1 on success when every opponent slot was rebuilt or has fainted (a full determinization).
int Gen3Search_Determinize(const struct Gen3DetMon *mons, int count, const struct Gen3DetHidden *hidden,
                           int64_t hiddenSeed);

// The random state of the turn, redrawn at a search root (legal and perfect modes):
//   gRngValue = seed; gRandomTurnNumber (this turn's Quick Claw roll) = Random();
//   the opponent's choice for this turn, already made by its AI while the player's controller
//   waits (action, move, target, switch-in), is undone so its AI chooses again from this state;
//   at a forced switch, the opponent's pending replacement is chosen again the same way.
void Gen3Search_RedrawTurn(uint32_t seed);

// At an ACTION decision of the live game: 1 if the player has a usable move (a move in the slot
// and not in `unusable`) or a switch target (canSwitch and a non-fainted party Pokemon 0-2 other
// than the active one), as BattleView.usable_moves / switch_targets. 0: nothing to choose.
int Gen3Search_HasChoice(uint8_t unusable, int canSwitch);

#ifdef __cplusplus
}
#endif

#endif
