#pragma once

// The player's observer and the battle encoder (encodings v3 and v4) in C++: the equivalent of
// pybattle/view.py BattleObserver (observe / rebase / fast_copy) followed by rl/encode.py battle()
// with VERSION = 3 or 4, including rl/damage.py battle_estimates (v3: its IV rule
// FIXED_IVS[min(challenge, 7)][battle == 6] and species quirk, the ones the v3 network was trained on;
// v4: any IV 0-31 at the level shown, docs/RL_DECISIONS.md §17). The version is a process-wide switch,
// set_encode_version (rl.encode.set_version calls it).
//
//   ObsMemory mem;  obs_init(mem, hint_type, hint_style);
//   at every player decision of the battle, in order:  obs_observe(mem, game, forced, unusable, can_switch);
//   encode_battle(mem, game, ctx, out);                // the observation at the last observed decision
//   search: copy the struct (= BattleObserver.fast_copy), obs_rebase() after a determinization.
//
// ObsMemory is trivially copyable. The game's battle RAM is read in src/gen3/observer_host.c; the
// rules (turn events, reveals, counters, the view, the encoding) live in src/gen3/observer.cpp and
// follow the Python code line by line (tests/python/test_observer_cpp.py checks they agree).

#include "gen3_game.hpp"

#include <cstdint>

namespace pkmn {

constexpr int OBS_MAX_REVEALED = 8;       // revealed moves kept per enemy Pokemon (4 in a real battle)

// A decoded struct BattlePokemon (pybattle/emu/decode.py BattleMon)
struct ObsBMon {
    int32_t species, stats[5], moves[4], ivs[6], ability_num, stat_stages[8], ability, types[2], pp[4];
    int32_t hp, level, max_hp, item;
    uint32_t status1, status2;
};

// A decoded party Pokemon (PartyMon), the fields the observer uses
struct ObsPMon {
    int32_t species, item, moves[4], pp[4], ability_num, level, hp, max_hp, stats[5];
    uint32_t status;
};

// _Snap: the battle RAM at a decision
struct ObsSnap {
    ObsBMon mons[2];
    int32_t idx[2];
    ObsPMon parties[2][3];
    uint8_t dis[2][0x1C];
    uint32_t status3[2];
    uint16_t sides[2];
    uint8_t timers[24];
    uint16_t weather;
    uint8_t wfk[44];
    int32_t last_moves[2], last_printed[2], bide[2], crit, locked[2];
    int32_t turn, last_used[2], player_switches, rnd_turn, order[2];
    bool forced;
};

// _Snap.key + the observe() arguments (the cache of BattleObserver.observe)
struct ObsCacheKey {
    bool valid;
    bool forced;
    uint8_t raw[0x58 * 2];
    uint8_t idx[4];
    int32_t turn;
    uint8_t last[4];
    int32_t rnd_turn;
    int32_t unusable;
    bool can_switch;
};

// TurnEvents
struct ObsEvents {
    int32_t turn, turns;
    bool in_progress;
    int32_t first, own_action, own_move, own_crit, enemy_action, enemy_move, enemy_crit;
    int32_t damage_dealt_pixels, damage_taken;
    bool own_fainted, enemy_fainted;
};

// ActiveState (the fields the encoder uses; bools / counters in rl/encode.py ACTIVE_BOOLS / ACTIVE_COUNTERS order)
struct ObsActive {
    int32_t party_index, stat_stages[7], types[2];
    bool bools[26];
    int32_t counters[16];
    int32_t perish_count, substitute_hp, semi_invulnerable;
    int32_t encored_move, disabled_move, locked_move, charging_move;
};

// OwnMon
struct ObsOwnMon {
    int32_t species, level, hp, max_hp, stats[5], moves[4], pp[4], item, ability, status, types[2], sleep_turns;
};

// SeenMon
struct ObsSeenMon {
    bool seen, fainted;
    int32_t species, level, hp_pixels, status;
    int32_t n_moves, moves[OBS_MAX_REVEALED];
    int32_t item;                       // revealed item, -1 = not revealed (None)
    int32_t n_types, types[2];
    int32_t n_base, base_stats[6];
    int32_t n_abilities, possible_abilities[2];
    int32_t revealed_ability, sleep_turns;
};

// SideState: reflect, light screen, safeguard, mist, spikes, future sight turns / move, wish
struct ObsSide {
    int32_t reflect_turns, light_screen_turns, safeguard_turns, mist_turns, spikes;
    int32_t future_sight_turns, future_sight_move, wish_turns;
};

// BattleView (what the encoder uses)
struct ObsView {
    bool valid;
    ObsOwnMon own[3];
    ObsSeenMon enemy[3];
    ObsActive own_active, enemy_active;
    int32_t weather;
    bool weather_permanent;
    int32_t weather_turns_left;
    ObsSide own_side, enemy_side;
    int32_t turn;
    bool forced_switch, must_struggle;
    bool usable[4];
    bool mask[7];                       // legal_actions: moves 0-3, switches to party 0-2
    ObsEvents last_turn;
    int32_t hint_type, hint_style;
};

struct ObsSleepRec { bool on; int32_t attempts, raw; };

// BattleObserver._vol[b]
struct ObsVol {
    bool has_mon;
    int32_t mon;
    bool has_conf;
    int32_t conf_tried, conf_raw;
    bool has[5];                        // encore, disable, wrapped, uproar, rampage
    int32_t start[5];
};

// The whole BattleObserver memory.
struct ObsMemory {
    int32_t hint_type, hint_style;
    uint8_t n_revealed[3];
    int32_t revealed_moves[3][OBS_MAX_REVEALED];
    int32_t revealed_items[3];          // -1: none
    int32_t revealed_abilities[3];      // 0: none
    uint8_t seen;                       // bit i: enemy party slot i seen
    int32_t last_enemy_item[3];         // -1: absent
    int32_t bench_status[3];            // -1: absent
    ObsSleepRec sleep[2][3];
    ObsVol vol[2];
    int32_t turns_done;
    bool has_start, has_prev;
    ObsSnap start, prev;
    ObsEvents events;
    ObsCacheKey cache;
    ObsView view;                       // the view of the last observed decision
    bool overflow;                      // a rule met a case the fixed-size memory cannot hold (never in practice)
};

void obs_init(ObsMemory& mem, uint16_t hint_type = 18, uint16_t hint_style = 0);
// BattleObserver.observe: call at every player decision of the battle, in order.
void obs_observe(ObsMemory& mem, Gen3Game& game, bool forced, uint8_t unusable_mask, bool can_switch);
// BattleObserver.rebase: the opponent's hidden state of `game` was rewritten (a determinization).
void obs_rebase(ObsMemory& mem, Gen3Game& game, bool forced);

// ---- encodings v3 / v4 (rl/encode.py) ----
struct EncodeCtx { int streak, battle, challenge, rents; };
// Layout sizes. MON_NUM depends on the version (v4 adds N_DEFEATED numbers per Pokemon token); the numeric arrays
// of EncodedObs have room for the largest layout (any version, also v2 through the Python encoder) and are packed:
// token i's numbers start at i * mon_w (move_num: (i * 4 + j) * move_w).
constexpr int MON_IDS = 19, CTX_IDS = 2, N_DEFEATED = 6;
constexpr int MON_NUM_V3 = 106, MON_NUM_V4 = MON_NUM_V3 + N_DEFEATED, MOVE_NUM = 17, CTX_NUM = 99;
constexpr int MON_NUM_MAX = MON_NUM_V4, MOVE_NUM_MAX = MOVE_NUM, CTX_NUM_MAX = CTX_NUM;
struct EncodedObs {
    int64_t mon_ids[6][MON_IDS];
    float mon_num[6 * MON_NUM_MAX];
    float move_num[6 * 4 * MOVE_NUM_MAX];
    int64_t ctx_ids[CTX_IDS];
    float ctx_num[CTX_NUM_MAX];
    bool mask[7];
    int64_t active;
    int32_t mon_w, move_w, ctx_w;       // this observation's MON_NUM, MOVE_NUM, CTX_NUM
};

// The encoding version of encode_battle / encode_view (3 or 4; default 3). Throws std::invalid_argument otherwise.
void set_encode_version(int version);
int encode_version();
int encode_mon_num(int version);        // MON_NUM of a version (3, 4)

// rl.encode.battle(view, ctx) of the last observed view (`game` is not read: the view is in the memory).
void encode_battle(const ObsMemory& mem, Gen3Game& game, const EncodeCtx& ctx, EncodedObs& out);
void encode_view(const ObsView& view, const EncodeCtx& ctx, EncodedObs& out);

}  // namespace pkmn
