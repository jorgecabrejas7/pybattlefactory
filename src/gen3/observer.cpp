// The player's observer and the v3 / v4 battle encoder in C++ (include/gen3_observer.hpp).
//
// Every function here mirrors a function of pybattle/view.py (BattleObserver), rl/encode.py or
// rl/damage.py, with the same name, and follows it line by line; read the Python for the reasons
// behind each rule. Numbers are computed in double like Python and rounded to float32 where numpy
// does it (tests/python/test_observer_cpp.py compares both on thousands of decisions).

#include "gen3_observer.hpp"

extern "C" {
#include "gen3/observer_host.h"
}
#include "gen3/generated/observer_tables.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

namespace pkmn {

namespace {

namespace A = obsgen::ability;
namespace E = obsgen::effect;
namespace H = obsgen::hold;
namespace TY = obsgen::type;

constexpr int N_TYPES = 18;
constexpr int TYPE_MYSTERY = 9;
constexpr int HP_BAR_PIXELS = 48;
constexpr int MOVE_UNAVAILABLE = 0xFFFF;
constexpr int NO_HINT_TYPE = 18;

// status1
constexpr uint32_t STATUS1_SLEEP = 0x7, STATUS1_POISON = 1 << 3, STATUS1_BURN = 1 << 4, STATUS1_FREEZE = 1 << 5,
                   STATUS1_PARALYSIS = 1 << 6, STATUS1_TOXIC = 1 << 7, STATUS1_TOXIC_COUNTER = 0xF << 8;
constexpr uint32_t STATUS1_ANY = STATUS1_SLEEP | STATUS1_POISON | STATUS1_BURN | STATUS1_FREEZE | STATUS1_PARALYSIS |
                                 STATUS1_TOXIC;
// status2
constexpr uint32_t STATUS2_CONFUSION = 0x7, STATUS2_UPROAR = 0x7 << 4, STATUS2_BIDE = 0x3 << 8,
                   STATUS2_LOCK_CONFUSE = 0x3 << 10, STATUS2_MULTIPLETURNS = 1 << 12, STATUS2_WRAPPED = 0x7 << 13,
                   STATUS2_INFATUATION = 0xF << 16, STATUS2_FOCUS_ENERGY = 1 << 20, STATUS2_TRANSFORMED = 1 << 21,
                   STATUS2_RECHARGE = 1 << 22, STATUS2_RAGE = 1 << 23, STATUS2_SUBSTITUTE = 1 << 24,
                   STATUS2_DESTINY_BOND = 1 << 25, STATUS2_ESCAPE_PREVENTION = 1 << 26, STATUS2_NIGHTMARE = 1 << 27,
                   STATUS2_CURSED = 1 << 28, STATUS2_FORESIGHT = 1 << 29, STATUS2_DEFENSE_CURL = 1u << 30,
                   STATUS2_TORMENT = 1u << 31;
// status3
constexpr uint32_t STATUS3_LEECHSEED_BATTLER = 0x3, STATUS3_LEECHSEED = 1 << 2, STATUS3_ALWAYS_HITS = 0x3 << 3,
                   STATUS3_PERISH_SONG = 1 << 5, STATUS3_ON_AIR = 1 << 6, STATUS3_UNDERGROUND = 1 << 7,
                   STATUS3_MINIMIZED = 1 << 8, STATUS3_CHARGED_UP = 1 << 9, STATUS3_ROOTED = 1 << 10,
                   STATUS3_YAWN = 0x3 << 11, STATUS3_IMPRISONED_OTHERS = 1 << 13, STATUS3_GRUDGE = 1 << 14,
                   STATUS3_MUDSPORT = 1 << 16, STATUS3_WATERSPORT = 1 << 17, STATUS3_UNDERWATER = 1 << 18,
                   STATUS3_INTIMIDATE_POKES = 1 << 19, STATUS3_TRACE = 1 << 20;
constexpr uint32_t STATUS3_SEMI_INVULNERABLE = STATUS3_ON_AIR | STATUS3_UNDERGROUND | STATUS3_UNDERWATER;
// side status
constexpr uint32_t SIDE_STATUS_REFLECT = 1 << 0, SIDE_STATUS_LIGHTSCREEN = 1 << 1, SIDE_STATUS_SAFEGUARD = 1 << 5,
                   SIDE_STATUS_MIST = 1 << 8;
// gBattleWeather
constexpr uint32_t B_WEATHER_RAIN = 0x7, B_WEATHER_RAIN_PERMANENT = 1 << 2, B_WEATHER_SANDSTORM = 0x3 << 3,
                   B_WEATHER_SANDSTORM_PERMANENT = 1 << 4, B_WEATHER_SUN = 0x3 << 5, B_WEATHER_SUN_PERMANENT = 1 << 6,
                   B_WEATHER_HAIL = 1 << 7;
constexpr int FLAG_MAKES_CONTACT = 1 << 0;

enum { WEATHER_NONE, WEATHER_RAIN, WEATHER_SUN, WEATHER_SANDSTORM, WEATHER_HAIL };
enum { ACTION_NONE, ACTION_MOVE, ACTION_SWITCH, ACTION_CANT_MOVE };
constexpr int FAINTED = -1;

const int ACC_RATIOS[13][2] = {{33, 100}, {36, 100}, {43, 100}, {50, 100}, {60, 100}, {75, 100}, {1, 1},
                               {133, 100}, {166, 100}, {2, 1}, {233, 100}, {133, 50}, {3, 1}};

// ---- static game knowledge ------------------------------------------------------------------------------------

struct Tables {
    const Gen3ObsTables* t;
    float typeEff[N_TYPES][N_TYPES];   // rl/gamedata.py TYPE_EFFECTIVENESS (last entry wins)
    bool immune[256][N_TYPES];         // view.py _type_immune: any (atk, def, 0) entry
    Tables() : t(Gen3Obs_Tables()) {
        for (auto& r : typeEff)
            for (float& x : r) x = 1.0f;
        std::memset(immune, 0, sizeof(immune));
        for (int i = 0; i + 2 < GEN3_OBS_TYPE_CHART; i += 3) {
            int a = t->typeChart[i], d = t->typeChart[i + 1], m = t->typeChart[i + 2];
            if (a < N_TYPES && d < N_TYPES) typeEff[a][d] = m / 10.0f;
            if (m == 0 && d < N_TYPES) immune[a][d] = true;
        }
    }
};

const Tables& tables() {
    static const Tables tb;
    return tb;
}

inline const Gen3ObsMove& MOVE(int m) {
    const Tables& tb = tables();
    return tb.t->moves[(unsigned)m < (unsigned)GEN3_OBS_MOVES_COUNT ? m : 0];
}
inline const Gen3ObsSpecies& SPECIES(int s) {
    return tables().t->species[(unsigned)s < (unsigned)GEN3_OBS_NUM_SPECIES ? s : 0];
}
inline int item_hold(int item) {
    return (0 < item && item < GEN3_OBS_ITEMS_COUNT) ? tables().t->items[item].holdEffect : 0;
}
inline int item_param(int item) {
    return (0 < item && item < GEN3_OBS_ITEMS_COUNT) ? tables().t->items[item].holdEffectParam : 0;
}
// damage.py uses ITEMS[atk_item] without the 0 < item guard (item 0 is ITEM_NONE: hold effect 0)
inline int item_hold_raw(int item) {
    return (unsigned)item < (unsigned)GEN3_OBS_ITEMS_COUNT ? tables().t->items[item].holdEffect : 0;
}
inline int item_param_raw(int item) {
    return (unsigned)item < (unsigned)GEN3_OBS_ITEMS_COUNT ? tables().t->items[item].holdEffectParam : 0;
}

inline bool in(int x, std::initializer_list<int> l) {
    for (int v : l)
        if (v == x) return true;
    return false;
}

int major_status(uint32_t s1) {
    if (s1 & STATUS1_SLEEP) return 1;
    if (s1 & STATUS1_TOXIC) return 6;
    if (s1 & STATUS1_POISON) return 2;
    if (s1 & STATUS1_BURN) return 3;
    if (s1 & STATUS1_FREEZE) return 4;
    if (s1 & STATUS1_PARALYSIS) return 5;
    return 0;
}

int hp_bar_pixels(int hp, int max_hp) {
    if (max_hp <= 0) return 0;
    int64_t v = (int64_t)hp * HP_BAR_PIXELS;
    int64_t q = v / max_hp;
    if ((v % max_hp != 0) && ((v < 0) != (max_hp < 0))) q--;      // floor division
    return (q == 0 && hp > 0) ? 1 : (int)q;
}

int ability_of(int species, int ability_num) {
    const auto& a = SPECIES(species).abilities;
    return (ability_num && a[1]) ? a[1] : a[0];
}

bool is_possible_ability(int species, int ability) {
    const auto& a = SPECIES(species).abilities;
    return ability == a[0] || (a[1] && a[1] != a[0] && ability == a[1]);
}

int priority_of(int move) { return (0 < move && move < GEN3_OBS_MOVES_COUNT) ? MOVE(move).priority : 0; }
int effect_of(int move) { return (0 < move && move < GEN3_OBS_MOVES_COUNT) ? MOVE(move).effect : 0; }
bool is(int move, std::initializer_list<int> effects) {
    return 0 < move && move < GEN3_OBS_MOVES_COUNT && in(MOVE(move).effect, effects);
}

#define TWO_TURN E::SOLAR_BEAM, E::RAZOR_WIND, E::SKY_ATTACK, E::SKULL_BASH, E::SEMI_INVULNERABLE
#define CALLERS E::METRONOME, E::SLEEP_TALK, E::ASSIST, E::MIRROR_MOVE, E::NATURE_POWER, E::SECRET_POWER

int move_type(int move, const ObsBMon& mon, int weather) {
    if (is(move, {E::HIDDEN_POWER})) {
        const int32_t* iv = mon.ivs;
        int bits = (iv[0] & 1) | (iv[1] & 1) << 1 | (iv[2] & 1) << 2 | (iv[3] & 1) << 3 | (iv[4] & 1) << 4 |
                   (iv[5] & 1) << 5;
        int t = 15 * bits / 63 + 1;
        return t >= TY::MYSTERY ? t + 1 : t;
    }
    if (is(move, {E::WEATHER_BALL}) && weather) {
        if (weather & B_WEATHER_RAIN) return TY::WATER;
        if (weather & B_WEATHER_SUN) return TY::FIRE;
        if (weather & B_WEATHER_SANDSTORM) return TY::ROCK;
        if (weather & B_WEATHER_HAIL) return TY::ICE;
    }
    return MOVE(move).type;
}

bool type_immune(int mtype, const int32_t types[2]) {
    if ((unsigned)mtype >= 256) return false;
    const Tables& tb = tables();
    for (int k = 0; k < 2; k++)
        if ((unsigned)types[k] < (unsigned)N_TYPES && tb.immune[mtype][types[k]]) return true;
    return false;
}

uint32_t status_class(uint32_t s1) {
    if (s1 & STATUS1_SLEEP) return STATUS1_SLEEP;
    if (s1 & (STATUS1_POISON | STATUS1_TOXIC)) return STATUS1_POISON;
    return s1 & (STATUS1_BURN | STATUS1_FREEZE | STATUS1_PARALYSIS);
}

// ---- snapshots ------------------------------------------------------------------------------------------------

inline uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline uint32_t rd32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

void decode_bmon(const uint8_t* d, ObsBMon& m) {
    m.species = rd16(d);
    for (int i = 0; i < 5; i++) m.stats[i] = rd16(d + 2 + 2 * i);
    for (int i = 0; i < 4; i++) m.moves[i] = rd16(d + 0x0C + 2 * i);
    uint32_t iv = rd32(d + 0x14);
    for (int i = 0; i < 6; i++) m.ivs[i] = (iv >> (5 * i)) & 31;
    m.ability_num = (iv >> 31) & 1;
    for (int i = 0; i < 8; i++) m.stat_stages[i] = d[0x18 + i];
    m.ability = d[0x20];
    m.types[0] = d[0x21];
    m.types[1] = d[0x22];
    for (int i = 0; i < 4; i++) m.pp[i] = d[0x24 + i];
    m.hp = rd16(d + 0x28);
    m.level = d[0x2A];
    m.max_hp = rd16(d + 0x2C);
    m.item = rd16(d + 0x2E);
    m.status1 = rd32(d + 0x4C);
    m.status2 = rd32(d + 0x50);
}

void read_snap(Gen3Game& game, bool forced, ObsSnap& s, ObsCacheKey* key) {
    Gen3ObsRaw r;
    game.makeLive();
    Gen3Obs_Read(&r);
    for (int b = 0; b < 2; b++) decode_bmon(r.mons[b], s.mons[b]);
    s.idx[0] = rd16(r.idx);
    s.idx[1] = rd16(r.idx + 2);
    for (int side = 0; side < 2; side++)
        for (int i = 0; i < 3; i++) {
            const Gen3ObsPartyMon& p = r.party[side][i];
            ObsPMon& o = s.parties[side][i];
            o.species = p.species;
            o.item = p.item;
            for (int k = 0; k < 4; k++) {
                o.moves[k] = p.moves[k];
                o.pp[k] = p.pp[k];
            }
            o.ability_num = p.abilityNum;
            o.level = p.level;
            o.hp = p.hp;
            o.max_hp = p.maxHp;
            for (int k = 0; k < 5; k++) o.stats[k] = p.stats[k];
            o.status = p.status;
        }
    std::memcpy(s.dis, r.dis, sizeof(s.dis));
    s.status3[0] = r.status3[0];
    s.status3[1] = r.status3[1];
    s.sides[0] = r.sides[0];
    s.sides[1] = r.sides[1];
    std::memcpy(s.timers, r.timers, sizeof(s.timers));
    s.weather = r.weather;
    std::memcpy(s.wfk, r.wfk, sizeof(s.wfk));
    s.last_moves[0] = rd16(r.lastMoves);
    s.last_moves[1] = rd16(r.lastMoves + 2);
    s.last_printed[0] = r.lastPrinted[0];
    s.last_printed[1] = r.lastPrinted[1];
    s.bide[0] = r.bide[0];
    s.bide[1] = r.bide[1];
    s.crit = r.crit;
    s.locked[0] = r.locked[0];
    s.locked[1] = r.locked[1];
    s.turn = r.results[0x13];
    s.last_used[0] = rd16(r.results + 0x22);
    s.last_used[1] = rd16(r.results + 0x24);
    s.player_switches = r.results[2];
    s.rnd_turn = r.rndTurn;
    s.order[0] = r.order[0];
    s.order[1] = r.order[1];
    s.forced = forced;
    if (key) {
        key->forced = forced;
        std::memcpy(key->raw, r.mons, sizeof(key->raw));
        std::memcpy(key->idx, r.idx, 4);
        key->turn = s.turn;
        std::memcpy(key->last, r.lastMoves, 4);
        key->rnd_turn = s.rnd_turn;
    }
}

bool same_snap_key(const ObsCacheKey& a, const ObsCacheKey& b) {
    return a.forced == b.forced && std::memcmp(a.raw, b.raw, sizeof(a.raw)) == 0 &&
           std::memcmp(a.idx, b.idx, 4) == 0 && a.turn == b.turn && std::memcmp(a.last, b.last, 4) == 0 &&
           a.rnd_turn == b.rnd_turn;
}

// disable-struct fields
inline int disabled_move(const ObsSnap& s, int b) { return rd16(s.dis[b] + 4); }
inline int encored_move(const ObsSnap& s, int b) { return rd16(s.dis[b] + 6); }
inline int protect_uses(const ObsSnap& s, int b) { return s.dis[b][8]; }
inline int stockpile(const ObsSnap& s, int b) { return s.dis[b][9]; }
inline int substitute_hp(const ObsSnap& s, int b) { return s.dis[b][0x0A]; }
inline int encore_timer(const ObsSnap& s, int b) { return s.dis[b][0x0E] & 0xF; }
inline int perish_timer(const ObsSnap& s, int b) { return s.dis[b][0x0F] & 0xF; }
inline int fury_cutter(const ObsSnap& s, int b) { return s.dis[b][0x10]; }
inline int rollout_timer(const ObsSnap& s, int b) { return s.dis[b][0x11] & 0xF; }
inline int charge_timer(const ObsSnap& s, int b) { return s.dis[b][0x12] & 0xF; }
inline int taunt_timer(const ObsSnap& s, int b) { return s.dis[b][0x13] & 0xF; }
inline int sure_hit_by(const ObsSnap& s, int b) { return s.dis[b][0x15]; }
inline int is_first_turn(const ObsSnap& s, int b) { return s.dis[b][0x16]; }
inline int spikes(const ObsSnap& s, int side) { return s.timers[12 * side + 10]; }

bool weather_has_effect(const ObsSnap& s) {
    for (int b = 0; b < 2; b++)
        if (s.mons[b].ability == A::CLOUD_NINE || s.mons[b].ability == A::AIR_LOCK) return false;
    return true;
}

inline const ObsPMon& pmon(const ObsSnap& s, int side, int i, ObsMemory& mem) {
    if ((unsigned)i >= 3u) {
        mem.overflow = true;
        i = 0;
    }
    return s.parties[side][i];
}

// _shift: `nw` (a rewritten Pokemon) as it was at the turn start
template <typename M>
M shift_mon(const M& nw, const M& start, const M& now, int M::*hp, int M::*max_hp) {
    M out = nw;
    int lost = start.*hp - now.*hp;
    if (lost && now.*max_hp) {
        double x = std::nearbyint((double)lost * nw.*max_hp / now.*max_hp);   // Python round(): half to even
        out.*hp = std::max(0, std::min(nw.*max_hp, nw.*hp + (int)x));
    }
    for (int k = 0; k < 4; k++) {
        int mv = nw.moves[k];
        if (!mv) continue;
        int j = -1;
        for (int q = 0; q < 4; q++)
            if (now.moves[q] == mv) {
                j = q;
                break;
            }
        if (j >= 0 && start.moves[j] == mv) out.pp[k] = std::max(0, nw.pp[k] + start.pp[j] - now.pp[j]);
    }
    return out;
}

// ---- the observer -------------------------------------------------------------------------------------------

struct Observer {
    ObsMemory& m;

    void reveal_move(int i, int move) {
        if ((unsigned)i >= 3u) {
            m.overflow = true;
            return;
        }
        for (int k = 0; k < m.n_revealed[i]; k++)
            if (m.revealed_moves[i][k] == move) return;
        if (m.n_revealed[i] >= OBS_MAX_REVEALED) {
            m.overflow = true;
            return;
        }
        m.revealed_moves[i][m.n_revealed[i]++] = move;
    }

    void reveal_ability(const ObsSnap& s, int i, int ability) {
        if ((unsigned)i >= 3u) return;
        if (ability && is_possible_ability(s.parties[1][i].species, ability)) m.revealed_abilities[i] = ability;
    }

    // --- turn events ---

    void side_action(const ObsSnap& a, const ObsSnap& s, int side, bool complete, int& act, int& move) {
        int m0 = a.idx[side], mn = s.idx[side];
        const ObsPMon& before = pmon(a, side, m0, m);
        const ObsPMon& after = pmon(s, side, m0, m);
        int used[4], nu = 0;
        for (int k = 0; k < 4; k++)
            if (after.moves[k] == before.moves[k] && after.pp[k] < before.pp[k]) used[nu++] = k;
        if (nu) {
            int last = mn == m0 ? s.last_moves[side] : 0;
            bool found = false;
            for (int q = 0; q < nu; q++)
                if (before.moves[used[q]] == last) found = true;
            act = ACTION_MOVE;
            move = found ? last : before.moves[used[nu - 1]];
            return;
        }
        if (mn == m0) {
            int lm = s.last_moves[side];
            if (lm != a.last_moves[side] || (complete && s.mons[side].hp > 0)) {
                if (lm == MOVE_UNAVAILABLE) {
                    act = ACTION_CANT_MOVE, move = 0;
                    return;
                }
                if (lm) {
                    act = ACTION_MOVE, move = lm;
                    return;
                }
            }
            if (complete && s.mons[side].hp == 0) {
                act = FAINTED, move = 0;
                return;
            }
            act = ACTION_NONE, move = 0;
            return;
        }
        if (after.hp == 0) {
            act = FAINTED, move = 0;
            return;
        }
        act = ACTION_SWITCH, move = 0;
    }

    static int64_t speed(const ObsSnap& a, int b) {
        const ObsBMon& mo = a.mons[b];
        int64_t mult = 1;
        if (weather_has_effect(a) && ((mo.ability == A::SWIFT_SWIM && (a.weather & B_WEATHER_RAIN)) ||
                                      (mo.ability == A::CHLOROPHYLL && (a.weather & B_WEATHER_SUN))))
            mult = 2;
        int st = std::max(0, std::min(12, mo.stat_stages[3]));
        int64_t spd = (int64_t)mo.stats[2] * mult * obsgen::STAT_STAGE_RATIOS[st][0] / obsgen::STAT_STAGE_RATIOS[st][1];
        int he = item_hold(mo.item), param = item_param(mo.item);
        if (he == H::MACHO_BRACE) spd /= 2;
        if (mo.status1 & STATUS1_PARALYSIS) spd /= 4;
        if (he == H::QUICK_CLAW && a.rnd_turn < (0xFFFF * param) / 100) spd = 0xFFFFFFFFll;
        return spd;
    }

    static int order(const ObsSnap& a, const int prio[2]) {
        if (prio[0] != prio[1]) return prio[0] > prio[1] ? 0 : 1;
        int64_t s0 = speed(a, 0), s1 = speed(a, 1);
        if (s0 == s1) return -1;
        return s0 > s1 ? 0 : 1;
    }

    int first(const ObsSnap& a, const ObsSnap& s, const ObsEvents& ev, int n_turns) {
        int own = ev.own_action, en = ev.enemy_action;
        if (own == ACTION_NONE && en == ACTION_NONE) return -1;
        if (own == ACTION_NONE) return 1;
        if (en == ACTION_NONE) return 0;
        if (own == ACTION_SWITCH) return 0;
        if (en == ACTION_SWITCH) return 1;
        if (n_turns > 1) return (s.order[0] == 0 || s.order[0] == 1) ? s.order[0] : -1;
        int prio[2] = {priority_of(s.last_used[0]), priority_of(s.last_used[1])};
        return order(a, prio);
    }

    struct Dmg { int lost, h0, max_hp; };

    void direct_damage(const ObsSnap& a, const ObsSnap& s, const ObsEvents& ev, Dmg out[2]) {
        int raw[2], h0[2], max_hp[2], costs[2] = {0, 0}, recoil_div[2] = {0, 0};
        bool damaged[2] = {true, true};
        for (int b = 0; b < 2; b++)
            raw[b] = s.bide[b] >= a.bide[b] ? std::max(0, s.bide[b] - a.bide[b]) : std::max(0, s.bide[b]);
        for (int side = 0; side < 2; side++) {
            int act = side == 0 ? ev.own_action : ev.enemy_action;
            int move = side == 0 ? ev.own_move : ev.enemy_move;
            bool passed = act == ACTION_MOVE && is(move, {E::BATON_PASS}) && ev.first == side;
            int victim = (act == ACTION_SWITCH || (passed && s.idx[side] != a.idx[side])) ? s.idx[side] : a.idx[side];
            const ObsPMon& pm = pmon(a, side, victim, m);
            max_hp[side] = pm.max_hp;
            h0[side] = victim == a.idx[side] ? a.mons[side].hp : pm.hp;
            const ObsBMon& user = a.mons[side];
            if (act == ACTION_MOVE) {
                if (is(move, {E::SUBSTITUTE}) && (s.mons[side].status2 & STATUS2_SUBSTITUTE) &&
                    !(user.status2 & STATUS2_SUBSTITUTE))
                    costs[side] += std::max(1, user.max_hp / 4);
                else if (is(move, {E::BELLY_DRUM}) && s.mons[side].stat_stages[1] == 12 && user.stat_stages[1] < 12)
                    costs[side] += user.max_hp / 2;
                else if (is(move, {E::CURSE}) && (user.types[0] == TY::GHOST || user.types[1] == TY::GHOST) &&
                         (s.mons[1 - side].status2 & STATUS2_CURSED) && !(a.mons[1 - side].status2 & STATUS2_CURSED))
                    costs[side] += user.max_hp / 2;
                else if (is(move, {E::PAIN_SPLIT}))
                    costs[side] += std::max(0, user.hp - (user.hp + a.mons[1 - side].hp) / 2);
                else if (is(move, {E::RECOIL, E::DOUBLE_EDGE}) && user.ability != A::ROCK_HEAD)
                    recoil_div[side] = is(move, {E::DOUBLE_EDGE}) ? 3 : 4;
            }
            if (victim != a.idx[side]) {
                int layers = spikes(a, side);
                const auto& sp = SPECIES(pm.species);
                if (layers && sp.types[0] != TY::FLYING && sp.types[1] != TY::FLYING &&
                    ability_of(pm.species, pm.ability_num) != A::LEVITATE) {
                    static const int div[3] = {8, 6, 4};
                    int spk = std::max(1, pm.max_hp / div[std::min(layers, 3) - 1]);
                    costs[side] += spk;
                    h0[side] = std::max(0, h0[side] - spk);
                }
            }
            int oact = side == 0 ? ev.enemy_action : ev.own_action;
            int omove = side == 0 ? ev.enemy_move : ev.own_move;
            bool charging = oact == ACTION_MOVE && (s.mons[1 - side].status2 & STATUS2_MULTIPLETURNS) &&
                            s.locked[1 - side] == omove && is(omove, {TWO_TURN}) &&
                            s.idx[1 - side] == a.idx[1 - side] && !(a.mons[1 - side].status2 & STATUS2_MULTIPLETURNS);
            damaged[side] = ev.turns != 1 || (oact == ACTION_MOVE && !charging &&
                                              (MOVE(omove).power || is(omove, {E::PAIN_SPLIT, CALLERS})));
        }
        int hit[2];
        for (int b = 0; b < 2; b++) hit[b] = std::max(0, raw[b] - costs[b]);
        for (int it = 0; it < 8; it++) {
            int rec[2];
            for (int b = 0; b < 2; b++) {
                int x = std::min(hit[1 - b], h0[1 - b]);
                rec[b] = (recoil_div[b] && x > 0) ? std::max(1, floordiv(x, recoil_div[b])) : 0;
            }
            for (int b = 0; b < 2; b++) hit[b] = std::max(0, raw[b] - costs[b] - rec[b]);
        }
        for (int b = 0; b < 2; b++) {
            int act = b == 0 ? ev.own_action : ev.enemy_action;
            int move = b == 0 ? ev.own_move : ev.enemy_move;
            if (act == ACTION_MOVE && ev.first == b) {
                int dealt = std::min(hit[1 - b], h0[1 - b]);
                int heal = 0;
                if (is(move, {E::ABSORB, E::DREAM_EATER}) && dealt)
                    heal += std::max(1, floordiv(dealt, 2));
                else if (is(move, {E::RESTORE_HP, E::SOFTBOILED, E::MORNING_SUN, E::SYNTHESIS, E::MOONLIGHT}))
                    heal += max_hp[b] / 2;
                else if (is(move, {E::REST}))
                    heal += max_hp[b];
                int it = a.mons[b].item;
                if (item_hold(it) == H::SHELL_BELL && dealt && MOVE(move).power) {
                    int p = item_param(it);
                    if (p) heal += std::max(1, floordiv(dealt, p));
                    else m.overflow = true;
                }
                h0[b] = std::min(max_hp[b], h0[b] + heal);
            }
        }
        for (int b = 0; b < 2; b++) out[b] = {damaged[b] ? std::min(hit[b], h0[b]) : 0, h0[b], max_hp[b]};
    }

    static int floordiv(int x, int y) {
        int q = x / y;
        if ((x % y != 0) && ((x < 0) != (y < 0))) q--;
        return q;
    }

    ObsEvents turn_events(const ObsSnap& a, const ObsSnap& s, int n_turns) {
        ObsEvents ev = default_events();
        ev.turn = a.turn;
        ev.turns = n_turns;
        ev.in_progress = s.forced && s.mons[0].hp > 0 && s.idx[0] == a.idx[0];
        int acts[2][2];
        for (int side = 0; side < 2; side++) side_action(a, s, side, !ev.in_progress, acts[side][0], acts[side][1]);
        int raw[2];
        for (int b = 0; b < 2; b++) raw[b] = s.bide[b] >= a.bide[b] ? s.bide[b] - a.bide[b] : s.bide[b];
        for (int side = 0; side < 2; side++) {
            if (acts[side][0] != FAINTED) continue;
            int oact = acts[1 - side][0], omove = acts[1 - side][1];
            int m0 = a.idx[side];
            bool ko_by_hit = raw[side] >= pmon(a, side, m0, m).hp;
            bool struck = raw[1 - side] > 0 && (oact == ACTION_MOVE || oact == ACTION_SWITCH) &&
                          !is(omove, {E::RECOIL, E::DOUBLE_EDGE, E::SUBSTITUTE, E::BELLY_DRUM, E::CURSE, E::PAIN_SPLIT});
            int prio[2];
            prio[side] = priority_of(s.last_used[side]);
            prio[1 - side] = priority_of(omove);
            bool opp_first = oact == ACTION_SWITCH || (oact == ACTION_MOVE && order(a, prio) == 1 - side);
            bool any_pp = false;
            for (int k = 0; k < 4; k++)
                if (a.mons[side].moves[k] && a.mons[side].pp[k]) any_pp = true;
            if (ko_by_hit && oact == ACTION_MOVE && MOVE(omove).power && opp_first && !struck)
                acts[side][0] = ACTION_NONE, acts[side][1] = 0;
            else if ((a.mons[side].status2 & STATUS2_MULTIPLETURNS) && a.locked[side])
                acts[side][0] = ACTION_MOVE, acts[side][1] = a.locked[side];
            else if (!any_pp)
                acts[side][0] = ACTION_MOVE, acts[side][1] = obsgen::move::STRUGGLE;
            else
                acts[side][0] = ACTION_CANT_MOVE, acts[side][1] = 0;
        }
        ev.own_action = acts[0][0], ev.own_move = acts[0][1];
        ev.enemy_action = acts[1][0], ev.enemy_move = acts[1][1];
        if (ev.own_action == ACTION_SWITCH && s.player_switches == a.player_switches) ev.own_action = ACTION_NONE;
        if (ev.enemy_action == ACTION_SWITCH && ev.own_action == ACTION_MOVE && is(ev.own_move, {E::ROAR}))
            ev.enemy_action = ACTION_NONE;
        for (int side = 0; side < 2; side++) {
            bool f = false;
            for (int i = 0; i < 3; i++)
                if (s.parties[side][i].hp == 0 && a.parties[side][i].hp > 0) f = true;
            (side == 0 ? ev.own_fainted : ev.enemy_fainted) = f;
        }
        ev.first = first(a, s, ev, n_turns);

        int movers[2], nm = 0;
        if (ev.own_action == ACTION_MOVE || ev.own_action == ACTION_CANT_MOVE) movers[nm++] = 0;
        if (ev.enemy_action == ACTION_MOVE || ev.enemy_action == ACTION_CANT_MOVE) movers[nm++] = 1;
        int last_mover = -2;
        if (nm == 1)
            last_mover = movers[0];
        else if (nm == 2 && (ev.first == 0 || ev.first == 1))
            last_mover = 1 - ev.first;
        Dmg dmg[2];
        direct_damage(a, s, ev, dmg);
        for (int side = 0; side < 2; side++) {
            int act = side == 0 ? ev.own_action : ev.enemy_action;
            int move = side == 0 ? ev.own_move : ev.enemy_move;
            int crit = -1;
            if (act != ACTION_MOVE || MOVE(move).power == 0)
                crit = 0;
            else if (n_turns == 1 && side == last_mover && s.crit == 2 &&
                     (s.bide[1 - side] != a.bide[1 - side] || (a.mons[1 - side].status2 & STATUS2_SUBSTITUTE)))
                crit = 1;
            else if (n_turns == 1 && side == last_mover && s.forced)
                crit = 0;
            (side == 0 ? ev.own_crit : ev.enemy_crit) = crit;
        }
        ev.damage_taken = dmg[0].lost;
        int h0 = dmg[1].h0, lost = dmg[1].lost;
        ev.damage_dealt_pixels = hp_bar_pixels(h0, dmg[1].max_hp) - hp_bar_pixels(std::max(0, h0 - lost), dmg[1].max_hp);
        return ev;
    }

    static ObsEvents default_events() {
        ObsEvents ev;
        std::memset(&ev, 0, sizeof(ev));
        ev.turn = -1;
        ev.first = -1;
        ev.own_crit = -1;
        ev.enemy_crit = -1;
        return ev;
    }

    // --- memory updates ---

    void update_seen(const ObsSnap& s) {
        const ObsSnap* prev = m.has_prev ? &m.prev : nullptr;
        if ((unsigned)s.idx[1] < 8u) m.seen |= (uint8_t)(1u << s.idx[1]);
        for (int i = 0; i < 3; i++) {
            const ObsPMon& mo = s.parties[1][i];
            if (prev) {
                const ObsPMon& pm = prev->parties[1][i];
                bool diff = mo.hp != pm.hp;
                for (int k = 0; k < 4; k++) diff = diff || mo.pp[k] != pm.pp[k];
                if (diff) m.seen |= (uint8_t)(1u << i);
                for (int k = 0; k < 4; k++)
                    if (mo.moves[k] && mo.moves[k] == pm.moves[k] && mo.pp[k] < pm.pp[k]) reveal_move(i, mo.moves[k]);
            }
        }
        int e = s.idx[1];
        int printed = s.last_printed[1];
        if (printed && printed != MOVE_UNAVAILABLE) {
            const ObsPMon& em = pmon(s, 1, e, m);
            if (in(printed, {em.moves[0], em.moves[1], em.moves[2], em.moves[3]})) reveal_move(e, printed);
        }
        if ((unsigned)e < 3u) {
            int prev_item = m.last_enemy_item[e];          // -1: absent (None)
            if (prev_item > 0 && s.mons[1].item != prev_item) m.revealed_items[e] = prev_item;
            m.last_enemy_item[e] = s.mons[1].item;
        } else {
            m.overflow = true;
        }
        if (prev && prev->idx[1] != e) {
            int i = prev->idx[1];
            if ((unsigned)i < 3u) m.bench_status[i] = major_status(prev->mons[1].status1);
            else m.overflow = true;
        }
    }

    bool our_move_surely_hit(const ObsSnap& a, const ObsSnap& s, const ObsEvents& ev, int move, bool target_start) {
        const ObsBMon& ours = a.mons[0];
        const ObsBMon* foe = target_start ? &a.mons[1] : nullptr;
        if (ev.enemy_action == ACTION_MOVE && is(ev.enemy_move, {E::PROTECT}) && protect_uses(s, 1) > 0) return false;
        if ((a.status3[1] & STATUS3_SEMI_INVULNERABLE) || (s.status3[1] & STATUS3_SEMI_INVULNERABLE)) return false;
        if (target_start && (a.status3[1] & STATUS3_ALWAYS_HITS) && sure_hit_by(a, 1) == 0) return true;
        if (is(move, {E::ALWAYS_HIT, E::VITAL_THROW})) return true;
        int acc = MOVE(move).accuracy;
        if (is(move, {E::THUNDER}) && weather_has_effect(a) && (a.weather & B_WEATHER_SUN)) acc = 50;
        if (is(move, {E::THUNDER}) && weather_has_effect(a) && (a.weather & B_WEATHER_RAIN)) return true;
        int eva = foe ? foe->stat_stages[7] : 6;
        int buff = (foe && (foe->status2 & STATUS2_FORESIGHT)) ? ours.stat_stages[6] : ours.stat_stages[6] + 6 - eva;
        buff = std::max(0, std::min(12, buff));
        int64_t calc = (int64_t)ACC_RATIOS[buff][0] * acc / ACC_RATIOS[buff][1];
        if (ours.ability == A::COMPOUND_EYES) calc = calc * 130 / 100;
        const ObsBMon& target = foe ? *foe : s.mons[1];
        if (weather_has_effect(a) && target.ability == A::SAND_VEIL && (a.weather & B_WEATHER_SANDSTORM))
            calc = calc * 80 / 100;
        if (ours.ability == A::HUSTLE && move_type(move, ours, a.weather) < TY::MYSTERY) calc = calc * 80 / 100;
        if (item_hold(target.item) == H::EVASION_UP) calc = calc * (100 - item_param(target.item)) / 100;
        return calc >= 100;
    }

    bool enemy_could_inflict(const ObsEvents& ev, uint32_t status) {
        if (ev.enemy_action != ACTION_MOVE) return false;
        int eff = effect_of(ev.enemy_move);
        switch (status) {
        case STATUS1_PARALYSIS:
            return in(eff, {E::PARALYZE, E::PARALYZE_HIT, E::TRI_ATTACK, E::THUNDER, CALLERS});
        case STATUS1_BURN:
            return in(eff, {E::BURN_HIT, E::WILL_O_WISP, E::TRI_ATTACK, E::BLAZE_KICK, CALLERS});
        case STATUS1_POISON:
            return in(eff, {E::POISON, E::POISON_HIT, E::TOXIC, E::POISON_FANG, E::TWINEEDLE, E::POISON_TAIL, CALLERS});
        case STATUS1_SLEEP:
            return in(eff, {E::SLEEP, E::YAWN, CALLERS});
        default:
            return false;
        }
    }

    static int stat_down(int eff) {
        if (eff == E::ATTACK_DOWN || eff == E::ATTACK_DOWN_2) return 1;
        if (eff == E::DEFENSE_DOWN || eff == E::DEFENSE_DOWN_2) return 2;
        if (eff == E::SPEED_DOWN || eff == E::SPEED_DOWN_2) return 3;
        if (eff == E::SPECIAL_ATTACK_DOWN || eff == E::SPECIAL_ATTACK_DOWN_2) return 4;
        if (eff == E::SPECIAL_DEFENSE_DOWN || eff == E::SPECIAL_DEFENSE_DOWN_2) return 5;
        if (eff == E::ACCURACY_DOWN || eff == E::ACCURACY_DOWN_2) return 6;
        if (eff == E::EVASION_DOWN || eff == E::EVASION_DOWN_2) return 7;
        return -1;
    }

    void reveal_by_our_move(const ObsSnap& a, const ObsSnap& s, const ObsEvents& ev) {
        if (ev.own_action != ACTION_MOVE || ev.turns != 1) return;
        int move = ev.own_move;
        int eff = effect_of(move);
        auto EF = [&](std::initializer_list<int> l) { return in(eff, l); };
        bool switched = ev.enemy_action == ACTION_SWITCH;
        int t = switched ? s.idx[1] : a.idx[1];
        const ObsBMon& tmon = (switched ? s : a).mons[1];
        const ObsPMon& tm = pmon(s, 1, t, m);
        int ab = ability_of(tm.species, tm.ability_num);
        if (tmon.ability != ab) return;
        const ObsBMon& ours = a.mons[0];
        int mtype = move_type(move, ours, a.weather);
        auto sure = [&]() { return our_move_surely_hit(a, s, ev, move, !switched); };
        bool dealt = s.bide[1] - a.bide[1] > 0;
        bool sub = tmon.status2 & STATUS2_SUBSTITUTE;
        auto reveal = [&]() { reveal_ability(s, t, ab); };
        int power = MOVE(move).power;
        if ((ab == A::WATER_ABSORB && mtype == TY::WATER && power) ||
            (ab == A::VOLT_ABSORB && mtype == TY::ELECTRIC && power) ||
            (ab == A::FLASH_FIRE && mtype == TY::FIRE && !(tmon.status1 & STATUS1_FREEZE))) {
            if (sure()) reveal();
            return;
        }
        if ((ab == A::LIMBER && EF({E::PARALYZE})) || (ab == A::IMMUNITY && EF({E::TOXIC, E::POISON})) ||
            (ab == A::OWN_TEMPO && EF({E::CONFUSE})) || (ab == A::SUCTION_CUPS && EF({E::ROAR})) ||
            ((ab == A::INSOMNIA || ab == A::VITAL_SPIRIT) && EF({E::YAWN}))) {
            reveal();
            return;
        }
        if ((ab == A::INSOMNIA || ab == A::VITAL_SPIRIT) && EF({E::SLEEP}) && !sub && !(tmon.status1 & STATUS1_SLEEP)) {
            reveal();
            return;
        }
        if (ab == A::WATER_VEIL && EF({E::WILL_O_WISP}) && !sub && !(tmon.status1 & STATUS1_BURN) &&
            tmon.types[0] != TY::FIRE && tmon.types[1] != TY::FIRE) {
            reveal();
            return;
        }
        if (ab == A::DAMP && EF({E::EXPLOSION}) && s.mons[0].hp > 0 && s.idx[0] == a.idx[0]) {
            reveal();
            return;
        }
        if (ab == A::STURDY && EF({E::OHKO}) && !type_immune(mtype, tmon.types) &&
            !(ev.enemy_action == ACTION_MOVE && is(ev.enemy_move, {E::PROTECT}) && protect_uses(s, 1) > 0) &&
            !((a.status3[1] | s.status3[1]) & STATUS3_SEMI_INVULNERABLE)) {
            reveal();
            return;
        }
        if (ab == A::OBLIVIOUS && EF({E::ATTRACT}) && sure()) {
            reveal();
            return;
        }
        int stat = stat_down(eff);
        if (stat >= 0 && !sub && !(s.mons[1].status2 & STATUS2_SUBSTITUTE) && !(s.sides[1] & SIDE_STATUS_MIST) &&
            sure()) {
            if (ab == A::CLEAR_BODY || ab == A::WHITE_SMOKE || (ab == A::KEEN_EYE && stat == 6) ||
                (ab == A::HYPER_CUTTER && stat == 1))
                reveal();
            return;
        }
        if (ab == A::LIQUID_OOZE && EF({E::ABSORB, E::DREAM_EATER}) && dealt) {
            reveal();
            return;
        }
        if (ab == A::STICKY_HOLD && tmon.item && pmon(s, 1, t, m).item == tmon.item &&
            ((EF({E::THIEF, E::KNOCK_OFF}) && dealt) || (EF({E::TRICK}) && !sub && sure()))) {
            reveal();
            return;
        }
        if (ab == A::INNER_FOCUS && EF({E::FAKE_OUT}) && dealt) {
            reveal();
            return;
        }
        if ((MOVE(move).flags & FLAG_MAKES_CONTACT) && power && dealt && s.idx[0] == a.idx[0] &&
            !(a.mons[0].status1 & STATUS1_ANY)) {
            uint32_t got = status_class(s.mons[0].status1);
            bool caused = false;
            if (ab == A::STATIC) caused = got == STATUS1_PARALYSIS;
            else if (ab == A::FLAME_BODY) caused = got == STATUS1_BURN;
            else if (ab == A::POISON_POINT) caused = got == STATUS1_POISON;
            else if (ab == A::EFFECT_SPORE)
                caused = got == STATUS1_POISON || got == STATUS1_PARALYSIS || got == STATUS1_SLEEP;
            if (got && caused && !enemy_could_inflict(ev, got) &&
                !(got == STATUS1_SLEEP && (a.status3[0] & STATUS3_YAWN))) {
                reveal();
                return;
            }
        }
        if (ab == A::SYNCHRONIZE && s.idx[0] == a.idx[0] && !(a.mons[0].status1 & STATUS1_ANY) &&
            !(tmon.status1 & STATUS1_ANY) && t == s.idx[1]) {
            uint32_t got_them = status_class(s.mons[1].status1);
            if ((got_them == STATUS1_POISON || got_them == STATUS1_BURN || got_them == STATUS1_PARALYSIS) &&
                status_class(s.mons[0].status1) == got_them && !enemy_could_inflict(ev, got_them))
                reveal();
        }
    }

    void reveal_end_of_turn(const ObsSnap& a, const ObsSnap& s, const ObsEvents& ev) {
        bool after_end = !ev.in_progress;
        int e = a.idx[1];
        if (!after_end || s.idx[1] != e || ev.enemy_action == ACTION_SWITCH || s.mons[1].hp == 0) return;
        const ObsBMon &em = s.mons[1], &m0 = a.mons[1];
        const ObsPMon& pm = pmon(s, 1, e, m);
        int ab = ability_of(pm.species, pm.ability_num);
        if (em.ability == ab && ab == A::SPEED_BOOST && m0.stat_stages[3] < 12) reveal_ability(s, e, ab);
        if (em.ability == ab && ab == A::LIQUID_OOZE && (s.status3[1] & STATUS3_LEECHSEED) &&
            (s.status3[1] & STATUS3_LEECHSEED_BATTLER) == 0 && s.mons[0].hp > 0)
            reveal_ability(s, e, ab);
        if (em.ability == ab && ab == A::SHED_SKIN && ev.turns == 1) {
            uint32_t had = m0.status1 & (STATUS1_PARALYSIS | STATUS1_BURN | STATUS1_POISON | STATUS1_TOXIC);
            bool cured_by_move = ev.enemy_action == ACTION_MOVE && is(ev.enemy_move, {E::REFRESH, E::REST, E::HEAL_BELL});
            if (had && !(em.status1 & STATUS1_ANY) && em.item == m0.item && !cured_by_move) reveal_ability(s, e, ab);
        }
        if (item_hold(em.item) == H::LEFTOVERS && em.hp < em.max_hp &&
            !(em.status1 & (STATUS1_POISON | STATUS1_TOXIC | STATUS1_BURN)) && !(s.status3[1] & STATUS3_LEECHSEED) &&
            !(em.status2 & (STATUS2_CURSED | STATUS2_NIGHTMARE | STATUS2_WRAPPED))) {
            if ((unsigned)e < 3u) m.revealed_items[e] = em.item;
        }
    }

    void update_reveals(const ObsSnap* a, const ObsSnap& s, const ObsEvents& ev, bool can_switch) {
        int e = s.idx[1];
        auto base = [&](int i) {
            const ObsPMon& p = pmon(s, 1, i, m);
            return ability_of(p.species, p.ability_num);
        };
        const ObsBMon& em = s.mons[1];
        if (em.ability == A::INTIMIDATE && A::INTIMIDATE == base(e) && !(s.status3[1] & STATUS3_INTIMIDATE_POKES))
            reveal_ability(s, e, A::INTIMIDATE);
        if (base(e) == A::TRACE && em.ability != A::TRACE && !(s.status3[1] & STATUS3_TRACE))
            reveal_ability(s, e, A::TRACE);
        if (!s.forced && !can_switch && !(s.mons[0].status2 & (STATUS2_WRAPPED | STATUS2_ESCAPE_PREVENTION)) &&
            !(s.status3[0] & STATUS3_ROOTED) && em.ability == base(e) &&
            (em.ability == A::SHADOW_TAG || em.ability == A::ARENA_TRAP || em.ability == A::MAGNET_PULL))
            reveal_ability(s, e, em.ability);
        const ObsSnap* prev = m.has_prev ? &m.prev : nullptr;
        if (s.mons[0].ability == A::INTIMIDATE && (!prev || prev->idx[0] != s.idx[0]) && (!prev || prev->idx[1] == e) &&
            em.ability == base(e) &&
            (em.ability == A::CLEAR_BODY || em.ability == A::HYPER_CUTTER || em.ability == A::WHITE_SMOKE))
            reveal_ability(s, e, em.ability);
        if (!a) return;
        reveal_by_our_move(*a, s, ev);
        reveal_end_of_turn(*a, s, ev);
    }

    void update_counters(const ObsSnap* a, const ObsSnap& s, const ObsEvents& ev) {
        for (int side = 0; side < 2; side++) {
            int ids[2] = {s.idx[side], a ? a->idx[side] : s.idx[side]};
            int n = ids[0] == ids[1] ? 1 : 2;
            for (int q = 0; q < n; q++) {
                int i = ids[q];
                if ((unsigned)i >= 3u) {
                    m.overflow = true;
                    continue;
                }
                if (i != s.idx[side] && s.parties[side][i].hp > 0) continue;
                uint32_t st = i == s.idx[side] ? s.mons[side].status1 : s.parties[side][i].status;
                int raw = st & STATUS1_SLEEP;
                ObsSleepRec& rec = m.sleep[side][i];
                if (!raw) {
                    rec.on = false;
                    continue;
                }
                if (!rec.on || raw > rec.raw) {
                    int tried = 0;
                    if (a && i == s.idx[side] && i == a->idx[side]) {
                        int act = side == 0 ? ev.own_action : ev.enemy_action;
                        if (act == ACTION_CANT_MOVE && ev.first == 1 - side && !(a->status3[side] & STATUS3_YAWN) &&
                            !(a->mons[side].status1 & STATUS1_ANY))
                            tried = 1;
                    }
                    rec = {true, tried, raw};
                } else if (raw < rec.raw) {
                    rec.attempts += 1;
                    rec.raw = raw;
                }
            }
        }
        for (int b = 0; b < 2; b++) {
            ObsVol& v = m.vol[b];
            if (!v.has_mon || v.mon != s.idx[b] || (m.has_prev && m.prev.idx[b] != s.idx[b])) {
                std::memset(&v, 0, sizeof(v));
                v.has_mon = true;
                v.mon = s.idx[b];
            }
            const ObsBMon& mo = s.mons[b];
            int conf = mo.status2 & STATUS2_CONFUSION;
            if (!conf) {
                v.has_conf = false;
            } else if (!v.has_conf || conf > v.conf_raw) {
                int tried = 0;
                if (a && s.idx[b] == a->idx[b]) {
                    int act = b == 0 ? ev.own_action : ev.enemy_action;
                    bool rampage_ended = (a->mons[b].status2 & STATUS2_LOCK_CONFUSE) && !(mo.status2 & STATUS2_LOCK_CONFUSE);
                    if ((act == ACTION_MOVE || act == ACTION_CANT_MOVE) && ev.first == 1 - b && !rampage_ended) tried = 1;
                }
                v.has_conf = true;
                v.conf_tried = tried;
                v.conf_raw = conf;
            } else if (conf < v.conf_raw) {
                v.conf_tried += 1;
                v.conf_raw = conf;
            }
            int since = m.turns_done - (s.forced ? 0 : 1);
            bool on[5] = {encored_move(s, b) && encore_timer(s, b), disabled_move(s, b) != 0,
                          (mo.status2 & STATUS2_WRAPPED) != 0, (mo.status2 & STATUS2_UPROAR) != 0,
                          (mo.status2 & STATUS2_LOCK_CONFUSE) != 0};
            for (int k = 0; k < 5; k++) {
                if (!on[k]) v.has[k] = false;
                else if (!v.has[k]) {
                    v.has[k] = true;
                    v.start[k] = since;
                }
            }
        }
    }

    int elapsed(int b, int k) const {
        return m.vol[b].has[k] ? std::max(0, m.turns_done - m.vol[b].start[k]) : 0;
    }

    // --- assembling the view ---

    void active(const ObsSnap& s, int b, ObsActive& out) const {
        const ObsBMon& mo = s.mons[b];
        uint32_t st2 = mo.status2, st3 = s.status3[b];
        bool locked = st2 & (STATUS2_MULTIPLETURNS | STATUS2_RECHARGE);
        int locked_move = locked ? s.locked[b] : 0;
        bool two_turn = locked_move && (st2 & STATUS2_MULTIPLETURNS) && is(locked_move, {TWO_TURN});
        int semi = (st3 & STATUS3_ON_AIR) ? 1 : (st3 & STATUS3_UNDERGROUND) ? 2 : (st3 & STATUS3_UNDERWATER) ? 3 : 0;
        const ObsVol& v = m.vol[b];
        out.party_index = s.idx[b];
        for (int i = 0; i < 7; i++) out.stat_stages[i] = mo.stat_stages[i + 1] - 6;
        out.types[0] = mo.types[0];
        out.types[1] = mo.types[1];
        bool* B = out.bools;
        B[0] = st2 & STATUS2_CONFUSION;
        B[1] = st2 & STATUS2_INFATUATION;
        B[2] = st2 & STATUS2_SUBSTITUTE;
        B[3] = st3 & STATUS3_LEECHSEED;
        B[4] = st2 & STATUS2_CURSED;
        B[5] = st2 & STATUS2_NIGHTMARE;
        B[6] = st2 & STATUS2_ESCAPE_PREVENTION;
        B[7] = st2 & STATUS2_FOCUS_ENERGY;
        B[8] = st2 & STATUS2_TRANSFORMED;
        B[9] = st3 & STATUS3_PERISH_SONG;
        B[10] = st3 & STATUS3_ROOTED;
        B[11] = st3 & STATUS3_YAWN;
        B[12] = st2 & STATUS2_TORMENT;
        B[13] = taunt_timer(s, b) > 0;
        B[14] = st2 & STATUS2_RECHARGE;
        B[15] = st2 & STATUS2_DESTINY_BOND;
        B[16] = st2 & STATUS2_DEFENSE_CURL;
        B[17] = st2 & STATUS2_FORESIGHT;
        B[18] = st3 & STATUS3_MINIMIZED;
        B[19] = st3 & STATUS3_CHARGED_UP;
        B[20] = st3 & STATUS3_IMPRISONED_OTHERS;
        B[21] = st3 & STATUS3_GRUDGE;
        B[22] = st3 & STATUS3_MUDSPORT;
        B[23] = st3 & STATUS3_WATERSPORT;
        B[24] = st2 & STATUS2_RAGE;
        B[25] = is_first_turn(s, b) > 0;
        int32_t* C = out.counters;
        C[0] = (v.has_conf && (st2 & STATUS2_CONFUSION)) ? v.conf_tried : 0;           // confusion_turns
        C[1] = (mo.status1 & STATUS1_TOXIC) ? (int)((mo.status1 & STATUS1_TOXIC_COUNTER) >> 8) : 0;
        C[2] = taunt_timer(s, b);
        C[3] = elapsed(b, 0);                                                          // encore_turns
        C[4] = elapsed(b, 1);                                                          // disable_turns
        C[5] = (int)((st3 & STATUS3_YAWN) >> 11);
        C[6] = elapsed(b, 2);                                                          // wrapped_turns
        C[7] = elapsed(b, 3);                                                          // uproar_turns
        C[8] = elapsed(b, 4);                                                          // rampage_turns
        C[9] = (int)((st2 & STATUS2_BIDE) >> 8);
        C[10] = rollout_timer(s, b);
        C[11] = fury_cutter(s, b);
        C[12] = stockpile(s, b);
        C[13] = (st3 & STATUS3_CHARGED_UP) ? charge_timer(s, b) : 0;
        C[14] = (int)((st3 & STATUS3_ALWAYS_HITS) >> 3);
        C[15] = protect_uses(s, b);
        out.perish_count = (st3 & STATUS3_PERISH_SONG) ? perish_timer(s, b) : -1;
        out.substitute_hp = (b == 0 && (st2 & STATUS2_SUBSTITUTE)) ? substitute_hp(s, b) : -1;
        out.semi_invulnerable = semi;
        out.encored_move = encored_move(s, b);
        out.disabled_move = disabled_move(s, b);
        out.locked_move = locked_move;
        out.charging_move = two_turn ? locked_move : 0;
    }

    static void side_state(const ObsSnap& s, int side, ObsSide& o) {
        const uint8_t* t = s.timers + 12 * side;
        uint16_t st = s.sides[side];
        int fs_counter = s.wfk[side], fs_move = rd16(s.wfk + 24 + 2 * side);
        o.reflect_turns = (st & SIDE_STATUS_REFLECT) ? t[0] : 0;
        o.light_screen_turns = (st & SIDE_STATUS_LIGHTSCREEN) ? t[2] : 0;
        o.mist_turns = (st & SIDE_STATUS_MIST) ? t[4] : 0;
        o.safeguard_turns = (st & SIDE_STATUS_SAFEGUARD) ? t[6] : 0;
        o.spikes = t[10];
        o.future_sight_turns = fs_counter;
        o.future_sight_move = fs_counter ? fs_move : 0;
        o.wish_turns = s.wfk[32 + side];
    }

    void build(const ObsSnap& s, const ObsEvents& ev, bool forced, int unusable, bool can_switch, ObsView& v) {
        std::memset(&v, 0, sizeof(v));
        v.valid = true;
        int a0 = s.idx[0];
        for (int i = 0; i < 3; i++) {
            const ObsPMon& p = s.parties[0][i];
            ObsOwnMon& o = v.own[i];
            o.species = p.species;
            o.level = p.level;
            o.hp = p.hp;
            o.max_hp = p.max_hp;
            for (int k = 0; k < 5; k++) o.stats[k] = p.stats[k];
            for (int k = 0; k < 4; k++) {
                o.moves[k] = p.moves[k];
                o.pp[k] = p.pp[k];
            }
            o.item = p.item;
            o.ability = ability_of(p.species, p.ability_num);
            o.status = major_status(p.status);
            o.types[0] = SPECIES(p.species).types[0];
            o.types[1] = SPECIES(p.species).types[1];
        }
        if ((unsigned)a0 < 3u) {
            ObsOwnMon& o = v.own[a0];
            o.hp = s.mons[0].hp;
            for (int k = 0; k < 4; k++) o.pp[k] = s.mons[0].pp[k];
            o.status = major_status(s.mons[0].status1);
        } else {
            m.overflow = true;
        }
        for (int i = 0; i < 3; i++) {
            const ObsSleepRec& rec = m.sleep[0][i];
            v.own[i].sleep_turns = (rec.on && v.own[i].status == 1) ? rec.attempts : 0;
        }

        int e = s.idx[1];
        for (int i = 0; i < 3; i++) {
            ObsSeenMon& sv = v.enemy[i];
            if (!(m.seen & (1u << i))) {
                sv.hp_pixels = -1;
                sv.item = -1;
                continue;
            }
            const ObsPMon& mo = s.parties[1][i];
            bool act = i == e;
            int hp = act ? s.mons[1].hp : mo.hp;
            int status = major_status(act ? s.mons[1].status1 : mo.status);
            if (!act && ability_of(mo.species, mo.ability_num) == A::NATURAL_CURE && hp > 0 && m.bench_status[i] >= 0)
                status = m.bench_status[i];
            if (hp == 0) status = 0;
            const ObsSleepRec& rec = m.sleep[1][i];
            sv.seen = true;
            sv.species = mo.species;
            sv.level = mo.level;
            sv.hp_pixels = hp_bar_pixels(hp, mo.max_hp);
            sv.status = status;
            sv.n_moves = m.n_revealed[i];
            for (int k = 0; k < m.n_revealed[i]; k++) sv.moves[k] = m.revealed_moves[i][k];
            sv.item = m.revealed_items[i];
            sv.fainted = hp == 0;
            sv.revealed_ability = m.revealed_abilities[i];
            sv.sleep_turns = (rec.on && status == 1) ? rec.attempts : 0;
            const auto& sp = SPECIES(mo.species);
            sv.n_types = 2;
            sv.types[0] = sp.types[0];
            sv.types[1] = sp.types[1];
            sv.n_base = 6;
            for (int k = 0; k < 6; k++) sv.base_stats[k] = sp.base[k];
            sv.possible_abilities[0] = sp.abilities[0];
            sv.n_abilities = 1;
            if (sp.abilities[1] && sp.abilities[1] != sp.abilities[0]) sv.possible_abilities[sv.n_abilities++] = sp.abilities[1];
        }

        int w = s.weather;
        int kind = WEATHER_NONE;
        if (w & B_WEATHER_RAIN) kind = WEATHER_RAIN;
        else if (w & B_WEATHER_SUN) kind = WEATHER_SUN;
        else if (w & B_WEATHER_SANDSTORM) kind = WEATHER_SANDSTORM;
        else if (w & B_WEATHER_HAIL) kind = WEATHER_HAIL;
        bool permanent = w & (B_WEATHER_RAIN_PERMANENT | B_WEATHER_SUN_PERMANENT | B_WEATHER_SANDSTORM_PERMANENT);
        v.weather = kind;
        v.weather_permanent = permanent;
        v.weather_turns_left = (kind && !permanent) ? s.wfk[40] : 0;

        bool switch_ok[3] = {false, false, false};
        if (forced || can_switch)
            for (int i = 0; i < 3; i++) switch_ok[i] = s.parties[0][i].hp > 0 && i != a0;
        bool any_usable = false;
        for (int i = 0; i < 4; i++) {
            v.usable[i] = !forced && s.mons[0].moves[i] && !((unusable >> i) & 1);
            any_usable = any_usable || v.usable[i];
        }
        v.must_struggle = !forced && !any_usable;
        if (!forced) {
            for (int i = 0; i < 4; i++) v.mask[i] = v.usable[i];
            if (!any_usable) v.mask[0] = true;
        }
        for (int i = 0; i < 3; i++) v.mask[4 + i] = switch_ok[i];

        active(s, 0, v.own_active);
        active(s, 1, v.enemy_active);
        side_state(s, 0, v.own_side);
        side_state(s, 1, v.enemy_side);
        v.turn = s.turn;
        v.forced_switch = forced;
        v.last_turn = ev;
        v.hint_type = m.hint_type;
        v.hint_style = m.hint_style;
    }

    void observe(Gen3Game& game, bool forced, int unusable, bool can_switch) {
        ObsSnap s;
        ObsCacheKey key;
        read_snap(game, forced, s, &key);
        key.unusable = unusable;
        key.can_switch = can_switch;
        key.valid = true;
        if (m.cache.valid && same_snap_key(key, m.cache) && key.unusable == m.cache.unusable &&
            key.can_switch == m.cache.can_switch)
            return;
        const ObsSnap* a = m.has_start ? &m.start : nullptr;
        bool new_turn_done = a && !forced;
        int n_turns = a ? std::max(1, s.turn - a->turn + (forced ? 1 : 0)) : 0;
        ObsEvents ev = a ? turn_events(*a, s, n_turns) : default_events();
        m.events = ev;
        if (new_turn_done) m.turns_done += std::max(1, s.turn - a->turn);

        update_seen(s);
        update_reveals(a, s, ev, can_switch);
        update_counters(a, s, ev);
        if (new_turn_done) record_turns(*a, s, ev);
        record_boosts(s);
        build(s, ev, forced, unusable, can_switch, m.view);

        if (!forced) {
            m.start = s;
            m.has_start = true;
        }
        m.prev = s;
        m.has_prev = true;
        m.cache = key;
    }

    // BattleObserver.finish
    void finish(Gen3Game& game) {
        if (m.finished || !m.has_start) return;
        m.finished = true;
        ObsSnap s;
        read_snap(game, false, s, nullptr);
        const ObsSnap& a = m.start;
        ObsEvents ev = turn_events(a, s, std::max(1, s.turn - a.turn));
        record_turns(a, s, ev);
        record_boosts(s);
    }

    // --- per-opponent records (BattleObserver._record_turns / _record_boosts) ---
    static void own_state(const ObsSnap& s, int i, int& hp, int& st) {
        if (i == s.idx[0]) {
            hp = s.mons[0].hp;
            st = major_status(s.mons[0].status1);
        } else {
            hp = s.parties[0][i].hp;
            st = major_status(s.parties[0][i].status);
        }
    }

    void record_turns(const ObsSnap& a, const ObsSnap& s, const ObsEvents& ev) {
        int j = ev.enemy_action == ACTION_SWITCH ? s.idx[1] : a.idx[1];
        if ((unsigned)j >= 3u) return;
        int32_t* rec = m.records[j];
        m.team_max_hp = 0;
        for (int i = 0; i < 3; i++) m.team_max_hp += s.parties[0][i].max_hp;
        bool rested = ev.own_action == ACTION_MOVE && is(ev.own_move, {E::REST});
        for (int i = 0; i < 3; i++) {
            int hp0, st0, hp1, st1;
            own_state(a, i, hp0, st0);
            own_state(s, i, hp1, st1);
            rec[0] += std::max(0, hp0 - hp1);
            if (hp0 > 0 && hp1 == 0) rec[1] += 1;
            if (st0 == 0 && st1 != 0 && hp1 > 0 && !(rested && i == a.idx[0] && st1 == 1)) rec[5] = 1;
        }
        rec[2] += ev.turns;
        if (ev.own_action == ACTION_MOVE && ev.damage_dealt_pixels > 0) rec[3] += 1;
    }

    void record_boosts(const ObsSnap& s) {
        int j = s.idx[1];
        if ((unsigned)j >= 3u) return;
        int boosts = 0;
        for (int k = 1; k < 8; k++) boosts += std::max(0, s.mons[1].stat_stages[k] - 6);
        if (boosts > m.records[j][4]) m.records[j][4] = boosts;
    }

    void rebase(Gen3Game& game, bool forced) {
        ObsSnap s;
        ObsCacheKey key;
        read_snap(game, forced, s, &key);
        if (!forced || !m.has_start) {
            m.start = s;
            m.has_start = true;
        } else if (m.has_prev) {
            const ObsSnap& a = m.start;
            const ObsSnap& old_prev = m.prev;
            ObsSnap a2 = a;
            for (int i = 0; i < 3; i++)
                a2.parties[1][i] = shift_mon(s.parties[1][i], a.parties[1][i], old_prev.parties[1][i], &ObsPMon::hp,
                                             &ObsPMon::max_hp);
            a2.mons[1] = a.idx[1] == s.idx[1]
                             ? shift_mon(s.mons[1], a.mons[1], old_prev.mons[1], &ObsBMon::hp, &ObsBMon::max_hp)
                             : a.mons[1];
            m.start = a2;
        }
        m.prev = s;
        m.has_prev = true;
        int e = s.idx[1];
        if ((unsigned)e < 3u && m.last_enemy_item[e] >= 0) m.last_enemy_item[e] = s.mons[1].item;
        for (int side = 0; side < 2; side++)
            for (int i = 0; i < 3; i++) {
                ObsSleepRec& rec = m.sleep[side][i];
                if (!rec.on) continue;
                uint32_t st = i == s.idx[side] ? s.mons[side].status1 : s.parties[side][i].status;
                if (st & STATUS1_SLEEP) rec.raw = st & STATUS1_SLEEP;
            }
        for (int b = 0; b < 2; b++) {
            ObsVol& v = m.vol[b];
            if (v.has_conf && (s.mons[b].status2 & STATUS2_CONFUSION)) v.conf_raw = s.mons[b].status2 & STATUS2_CONFUSION;
        }
        if (m.cache.valid) {
            int unusable = m.cache.unusable;
            bool cs = m.cache.can_switch;
            m.cache = key;
            m.cache.valid = true;
            m.cache.unusable = unusable;
            m.cache.can_switch = cs;
        }
    }
};

// ---- rl/damage.py ----------------------------------------------------------------------------------------------

double stage_mult(int stage) {
    int i = std::max(0, std::min(12, stage + 6));
    return (double)obsgen::STAT_STAGE_RATIOS[i][0] / obsgen::STAT_STAGE_RATIOS[i][1];
}

struct Range { double lo, hi; };

Range stat_range(int base, int iv, int level = 100) {
    double lo = std::floor((std::floor((double)((2 * base + iv) * level) / 100) + 5) * 0.9);
    double hi = std::floor((std::floor((double)((2 * base + iv + 63) * level) / 100) + 5) * 1.1);
    return {lo, hi};
}

Range hp_range(int base, int iv, int level = 100) {
    if (base == 1) return {1, 1};
    return {std::floor((double)((2 * base + iv) * level) / 100) + level + 10,
            std::floor((double)((2 * base + iv + 63) * level) / 100) + level + 10};
}

// v4: any IV 0-31 (and EVs 0-252, natures 0.9-1.1)
Range stat_range_any_iv(int base, int level) {
    double lo = std::floor((std::floor((double)(2 * base * level) / 100) + 5) * 0.9);
    double hi = std::floor((std::floor((double)((2 * base + 31 + 63) * level) / 100) + 5) * 1.1);
    return {lo, hi};
}

Range hp_range_any_iv(int base, int level) {
    if (base == 1) return {1, 1};
    return {std::floor((double)(2 * base * level) / 100) + level + 10,
            std::floor((double)((2 * base + 31 + 63) * level) / 100) + level + 10};
}

inline bool is_physical(int t) { return t < 9; }

// ability sets: 0 = empty set (ability 0 never matches the abilities tested)
int known_ability(const ObsSeenMon& foe) {
    if (foe.revealed_ability) return foe.revealed_ability;
    int ab[2], n = 0;
    for (int k = 0; k < foe.n_abilities; k++)
        if (foe.possible_abilities[k]) ab[n++] = foe.possible_abilities[k];
    if (n == 1 || (n == 2 && ab[0] == ab[1])) return ab[0];
    return 0;
}

double effectiveness(int mtype, const int32_t* def_types, int n_def) {
    double mult = 1.0;
    const Tables& tb = tables();
    for (int k = 0; k < n_def; k++) {
        int t = def_types[k];
        bool dup = false;
        for (int q = 0; q < k; q++) dup = dup || def_types[q] == t;
        if (dup || t >= N_TYPES || t < 0) continue;
        mult *= (unsigned)mtype < (unsigned)N_TYPES ? (double)tb.typeEff[mtype][t] : 1.0;
    }
    return mult;
}

struct DmgArgs {
    const int32_t* atk_types;
    int n_atk_types;
    const int32_t* def_types;
    int n_def_types;
    int atk_stage = 0, def_stage = 0, atk_item = 0, atk_ability = 0, def_ability = 0, atk_species = 0;
    bool atk_burned = false, screen = false;
    int weather = 0;
    double atk_hp_frac = 1.0;
    Range def_hp{1, 1};
    int version = 3;                    // <= 3: the v3 species quirk of Thick Club / Light Ball
};

// rl/damage.py's SP_CUBONE, SP_MAROWAK, SP_PIKACHU: the three species ids in id order (a known quirk of the v3
// features: Thick Club counts for Pikachu and Cubone, Light Ball for Marowak), kept as the network was trained.
constexpr int SP3_MIN = std::min({obsgen::species::CUBONE, obsgen::species::MAROWAK, obsgen::species::PIKACHU});
constexpr int SP3_MAX = std::max({obsgen::species::CUBONE, obsgen::species::MAROWAK, obsgen::species::PIKACHU});
constexpr int SP_CUBONE = SP3_MIN;
constexpr int SP_MAROWAK = obsgen::species::CUBONE + obsgen::species::MAROWAK + obsgen::species::PIKACHU - SP3_MIN - SP3_MAX;
constexpr int SP_PIKACHU = SP3_MAX;

int variable_power(int eff) {
    if (eff == E::RETURN || eff == E::FRUSTRATION) return 102;
    if (eff == E::LOW_KICK) return 60;
    if (eff == E::MAGNITUDE) return 71;
    if (eff == E::PRESENT) return 52;
    if (eff == E::HIDDEN_POWER) return 70;
    return 60;
}

Range move_damage(int move, Range atk, Range dfn, const DmgArgs& d) {
    if (!move) return {0, 0};
    const Gen3ObsMove& mv = MOVE(move);
    int eff = mv.effect, mtype = mv.type;
    if (in(eff, {E::COUNTER, E::MIRROR_COAT, E::OHKO, E::ENDEAVOR})) return {0, 0};
    double type_mult = effectiveness(mtype, d.def_types, d.n_def_types);
    if (type_mult == 0) return {0, 0};
    int da = d.def_ability;
    if ((mtype == TY::GROUND && da == A::LEVITATE) || (mtype == TY::WATER && da == A::WATER_ABSORB) ||
        (mtype == TY::ELECTRIC && da == A::VOLT_ABSORB) || (mtype == TY::FIRE && da == A::FLASH_FIRE) ||
        (da == A::WONDER_GUARD && type_mult <= 1))
        return {0, 0};
    if (eff == E::LEVEL_DAMAGE || eff == E::PSYWAVE) return {100, 100};
    if (eff == E::DRAGON_RAGE) return {40, 40};
    if (eff == E::SONICBOOM) return {20, 20};
    if (eff == E::SUPER_FANG) return {d.def_hp.lo / 2, d.def_hp.hi / 2};
    double power = mv.power;
    if (mv.power == 0) return {0, 0};
    if (mv.power == 1) power = variable_power(eff);
    double f = d.atk_hp_frac;
    if (eff == E::FLAIL)
        power = f < 0.04 ? 200 : f < 0.1 ? 150 : f < 0.2 ? 100 : f < 0.35 ? 80 : f < 0.69 ? 40 : 20;
    if (eff == E::ERUPTION) power = std::max(1.0, std::trunc(150 * f));
    if (eff == E::FACADE && d.atk_burned) power *= 2;
    bool phys = is_physical(mtype);
    double a_mult = stage_mult(d.atk_stage);
    int aa = d.atk_ability;
    if (phys && (aa == A::HUGE_POWER || aa == A::PURE_POWER)) a_mult *= 2;
    if (phys && aa == A::HUSTLE) a_mult *= 1.5;
    int he = item_hold_raw(d.atk_item);
    if (phys && he == H::CHOICE_BAND) a_mult *= 1.5;
    if (d.version <= 3) {
        if (phys && he == H::THICK_CLUB && (d.atk_species == SP_CUBONE || d.atk_species == SP_MAROWAK)) a_mult *= 2;
        if (!phys && he == H::LIGHT_BALL && d.atk_species == SP_PIKACHU) a_mult *= 2;
    } else {
        namespace SP = obsgen::species;
        if (phys && he == H::THICK_CLUB && (d.atk_species == SP::CUBONE || d.atk_species == SP::MAROWAK)) a_mult *= 2;
        if (!phys && he == H::LIGHT_BALL && d.atk_species == SP::PIKACHU) a_mult *= 2;
    }
    for (int k = 0; k < obsgen::N_TYPE_POWER_ITEMS; k++)
        if (obsgen::TYPE_POWER_ITEMS[k][0] == he) {
            if (obsgen::TYPE_POWER_ITEMS[k][1] == mtype) power = power * (100 + item_param_raw(d.atk_item)) / 100;
            break;
        }
    if (d.def_ability == A::THICK_FAT && (mtype == TY::FIRE || mtype == TY::ICE)) power /= 2;
    double d_mult = stage_mult(d.def_stage);
    int hits = (eff == E::DOUBLE_HIT || eff == E::TWINEEDLE) ? 2 : (eff == E::MULTI_HIT || eff == E::TRIPLE_KICK) ? 3 : 1;
    bool stab = false;
    for (int k = 0; k < d.n_atk_types; k++) stab = stab || d.atk_types[k] == mtype;
    double out[2];
    const double as[2] = {atk.lo, atk.hi}, ds[2] = {dfn.hi, dfn.lo}, rolls[2] = {0.85, 1.0};
    for (int r = 0; r < 2; r++) {
        double dd = ds[r] * d_mult;
        double den = dd < 1 ? 1.0 : dd;                // max(1, d * d_mult)
        double dmg = std::floor(std::floor(42 * power * as[r] * a_mult / den) / 50);
        if (phys && d.atk_burned && aa != A::GUTS) dmg = std::floor(dmg / 2);
        if (d.screen) dmg = std::floor(dmg / 2);
        dmg += 2;
        if (d.weather == WEATHER_RAIN)
            dmg *= mtype == TY::WATER ? 1.5 : mtype == TY::FIRE ? 0.5 : 1;
        else if (d.weather == WEATHER_SUN)
            dmg *= mtype == TY::FIRE ? 1.5 : mtype == TY::WATER ? 0.5 : 1;
        if (stab) dmg *= 1.5;
        dmg *= type_mult * rolls[r] * hits;
        out[r] = dmg;
    }
    return {out[0], out[1]};
}

struct Estimates {
    double own[3][4][3];
    int n_foe;
    double foe[4][3];
    double threat[3][2];
    double speed[3][2];
};

void battle_estimates(const ObsView& v, const EncodeCtx& ctx, int version, Estimates& es) {
    const ObsSeenMon& foe = v.enemy[std::max(0, std::min(2, v.enemy_active.party_index))];
    const ObsActive& fa = v.enemy_active;
    int base[6] = {0, 0, 0, 0, 0, 0};
    if (foe.species)
        for (int k = 0; k < 6; k++) base[k] = SPECIES(foe.species).base[k];
    Range f_hp, f_atk[2], f_def[2], f_spe;
    if (version <= 3) {
        int row = std::min(ctx.challenge, 7);
        row = std::max(0, std::min(obsgen::N_FIXED_IV_ROWS - 1, row));
        int iv = obsgen::FACTORY_FIXED_IVS[row][ctx.battle == 6 ? 1 : 0];
        f_hp = hp_range(base[0], iv);
        f_atk[0] = stat_range(base[1], iv);
        f_atk[1] = stat_range(base[4], iv);
        f_def[0] = stat_range(base[2], iv);
        f_def[1] = stat_range(base[5], iv);
        f_spe = stat_range(base[3], iv);
    } else {
        int lvl = foe.level ? foe.level : 100;
        f_hp = hp_range_any_iv(base[0], lvl);
        f_atk[0] = stat_range_any_iv(base[1], lvl);
        f_atk[1] = stat_range_any_iv(base[4], lvl);
        f_def[0] = stat_range_any_iv(base[2], lvl);
        f_def[1] = stat_range_any_iv(base[5], lvl);
        f_spe = stat_range_any_iv(base[3], lvl);
    }
    int f_ab = known_ability(foe);
    int f_item = foe.item > 0 ? foe.item : 0;
    double f_frac = std::max(foe.hp_pixels, 0) / 48.0;
    int a_own = v.own_active.party_index;
    static const int OWN_ATK[2] = {0, 3}, OWN_DEF[2] = {1, 4};
    for (int i = 0; i < 3; i++) {
        const ObsOwnMon& me = v.own[i];
        bool act = i == a_own;
        const ObsActive* st = act ? &v.own_active : nullptr;
        for (int j = 0; j < 4; j++) {
            int mv = me.moves[j];
            bool phys = mv && is_physical(MOVE(mv).type);
            int k = phys ? 0 : 1;
            double a = me.stats[OWN_ATK[k]];
            DmgArgs d;
            d.atk_types = me.types;
            d.n_atk_types = 2;
            d.def_types = fa.types;
            d.n_def_types = 2;
            d.atk_stage = st ? st->stat_stages[phys ? 0 : 3] : 0;
            d.def_stage = fa.stat_stages[phys ? 1 : 4];
            d.atk_item = me.item;
            d.atk_ability = me.ability;
            d.def_ability = f_ab;
            d.atk_species = me.species;
            d.atk_burned = me.status == 3;
            d.screen = (phys ? v.enemy_side.reflect_turns : v.enemy_side.light_screen_turns) > 0;
            d.weather = v.weather;
            d.atk_hp_frac = me.max_hp ? (double)me.hp / me.max_hp : 0;
            d.def_hp = f_hp;
            d.version = version;
            Range r = move_damage(mv, {a, a}, f_def[k], d);
            double lo_f = r.lo / f_hp.hi, hi_f = r.hi / f_hp.lo;
            es.own[i][j][0] = std::min(lo_f, 1.5);
            es.own[i][j][1] = std::min(hi_f, 1.5);
            es.own[i][j][2] = (hi_f >= f_frac && f_frac > 0) ? 1.0 : 0.0;
        }
        double worst = 0.0, ko = 0.0;
        for (int j = 0; j < foe.n_moves; j++) {
            int mv = foe.moves[j];
            bool phys = is_physical(MOVE(mv).type);
            int k = phys ? 0 : 1;
            double dv = me.stats[OWN_DEF[k]];
            DmgArgs d;
            d.atk_types = fa.types;
            d.n_atk_types = foe.seen ? 2 : 0;
            d.def_types = me.types;
            d.n_def_types = 2;
            d.atk_stage = fa.stat_stages[phys ? 0 : 3];
            d.def_stage = st ? st->stat_stages[phys ? 1 : 4] : 0;
            d.atk_item = f_item;
            d.atk_ability = f_ab;
            d.def_ability = me.ability;
            d.atk_species = foe.species;
            d.atk_burned = foe.status == 3;
            d.screen = (phys ? v.own_side.reflect_turns : v.own_side.light_screen_turns) > 0;
            d.weather = v.weather;
            d.atk_hp_frac = f_frac;
            d.def_hp = {(double)me.max_hp, (double)me.max_hp};
            d.version = version;
            Range r = move_damage(mv, f_atk[k], {dv, dv}, d);
            double frac = me.max_hp ? r.hi / me.max_hp : 0;
            worst = std::max(worst, std::min(frac, 1.5));
            ko = std::max(ko, (me.hp > 0 && r.hi >= me.hp) ? 1.0 : 0.0);
        }
        es.threat[i][0] = worst;
        es.threat[i][1] = ko;
        double spe = me.stats[2] * (act ? stage_mult(v.own_active.stat_stages[2]) : 1.0);
        double f_lo = f_spe.lo * stage_mult(fa.stat_stages[2]);
        double f_hi = f_spe.hi * stage_mult(fa.stat_stages[2]);
        es.speed[i][0] = spe > f_hi ? 1.0 : 0.0;
        es.speed[i][1] = (f_lo < spe && spe <= f_hi) ? 1.0 : 0.0;
    }
    const ObsOwnMon& me = v.own[std::max(0, std::min(2, a_own))];
    const ObsActive& st = v.own_active;
    es.n_foe = std::min(foe.n_moves, 4);
    for (int j = 0; j < es.n_foe; j++) {
        int mv = foe.moves[j];
        bool phys = is_physical(MOVE(mv).type);
        int k = phys ? 0 : 1;
        double dv = me.stats[OWN_DEF[k]];
        DmgArgs d;
        d.atk_types = fa.types;
        d.n_atk_types = 2;
        d.def_types = me.types;
        d.n_def_types = 2;
        d.atk_stage = fa.stat_stages[phys ? 0 : 3];
        d.def_stage = st.stat_stages[phys ? 1 : 4];
        d.atk_item = f_item;
        d.atk_ability = f_ab;
        d.def_ability = me.ability;
        d.atk_species = foe.species;
        d.atk_burned = foe.status == 3;
        d.screen = (phys ? v.own_side.reflect_turns : v.own_side.light_screen_turns) > 0;
        d.weather = v.weather;
        d.atk_hp_frac = f_frac;
        d.def_hp = {(double)me.max_hp, (double)me.max_hp};
        d.version = version;
        Range r = move_damage(mv, f_atk[k], {dv, dv}, d);
        es.foe[j][0] = std::min(r.lo / me.max_hp, 1.5);
        es.foe[j][1] = std::min(r.hi / me.max_hp, 1.5);
        es.foe[j][2] = (me.hp > 0 && r.hi >= me.hp) ? 1.0 : 0.0;
    }
}

// ---- rl/encode.py battle() -------------------------------------------------------------------------------------

constexpr int SPECIES_UNK = obsgen::N_SPECIES;
constexpr int MOVE_UNK = obsgen::N_MOVES;
constexpr int EFFECT_NONE = obsgen::N_EFFECTS, EFFECT_UNK = obsgen::N_EFFECTS + 1;
constexpr int TYPE_NONE = obsgen::N_TYPES, TYPE_UNK = obsgen::N_TYPES + 1;
constexpr int ITEM_UNK = obsgen::N_ITEMS;
constexpr int ABILITY_UNK = obsgen::N_ABILITIES;
constexpr int N_ACTIVE = 73;
constexpr int N_HINT_TYPES = N_TYPES + 1, N_STYLES = 8, N_ROUNDS = 7;
constexpr int ACTIVE_COUNTER_CAP[16] = {5, 15, 2, 6, 5, 2, 6, 5, 3, 2, 5, 5, 3, 2, 2, 4};
const int RENT_RANKS[5] = {15, 22, 29, 36, 43};
static_assert(7 + 26 + 16 + 2 + 4 + N_TYPES == N_ACTIVE, "N_ACTIVE");
static_assert(6 + 1 + 1 + 5 + 6 + 7 + 1 + 1 + 1 + N_ACTIVE + 4 == MON_NUM_V3, "MON_NUM");

int g_encode_version = 3;

struct Writer {
    float* p;
    void put(double x) { *p++ = (float)x; }
    void onehot(int i, int n) {
        for (int k = 0; k < n; k++) p[k] = 0.0f;
        if (0 <= i && i < n) p[i] = 1.0f;
        p += n;
    }
};

void types_ids(const int32_t* types, int n, int64_t& t1, int64_t& t2) {
    if (n == 0) {
        t1 = t2 = TYPE_UNK;
        return;
    }
    t1 = types[0];
    int64_t b = n > 1 ? types[1] : types[0];
    t2 = b != t1 ? b : TYPE_NONE;
}

double log_stage(int s) {
    return std::log2(s >= 0 ? (2.0 + s) / 2.0 : 2.0 / (2.0 - s)) / 2;
}

void active_feats(Writer& w, const ObsActive* a, int max_hp) {
    if (!a) {
        for (int k = 0; k < N_ACTIVE; k++) w.put(0.0);
        return;
    }
    for (int k = 0; k < 7; k++) w.put(log_stage(a->stat_stages[k]));
    for (int k = 0; k < 26; k++) w.put(a->bools[k] ? 1.0 : 0.0);
    for (int k = 0; k < 16; k++) w.put((double)std::min(a->counters[k], ACTIVE_COUNTER_CAP[k]) / ACTIVE_COUNTER_CAP[k]);
    w.put(a->perish_count >= 0 ? (a->perish_count + 1) / 4.0 : 0.0);
    w.put((max_hp && a->substitute_hp >= 0) ? (double)a->substitute_hp / max_hp : 0.0);
    w.onehot(a->semi_invulnerable, 4);
    float* types = w.p;
    for (int k = 0; k < N_TYPES; k++) types[k] = 0.0f;
    for (int k = 0; k < 2; k++)
        if (a->types[k] >= 0 && a->types[k] < N_TYPES) types[a->types[k]] = 1.0f;
    w.p += N_TYPES;
}

// _move_num
void move_num(float* out, int move, bool known, int pp, bool has_pp, bool usable, const bool flags[4], bool last,
              const double* est) {
    Writer w{out};
    if (!known || !move) {
        w.put(known ? 1.0 : 0.0);
        for (int k = 1; k < MOVE_NUM; k++) w.put(0.0);
        return;
    }
    const Gen3ObsMove& d = MOVE(move);
    int acc = d.accuracy;
    int base_pp = std::max((int)d.pp, 1);
    w.put(1.0);
    w.put(d.power / 255.0);
    w.put(acc / 100.0);
    w.put(acc == 0 ? 1.0 : 0.0);
    w.put(has_pp ? (double)pp / base_pp : 1.0);
    w.put(base_pp / 64.0);
    w.put(d.priority / 6.0);
    w.put(d.secondaryChance / 100.0);
    w.put(usable ? 1.0 : 0.0);
    for (int k = 0; k < 4; k++) w.put(flags[k] ? 1.0 : 0.0);
    w.put(last ? 1.0 : 0.0);
    for (int k = 0; k < 3; k++) w.put(est ? est[k] : 0.0);
}

void own_mon(const ObsOwnMon& m, const ObsActive* active, bool is_active, const bool* usable, int last_move,
             const double (*est)[3], const double* threat, const double* speed, int64_t* ids, float* num,
             float (*moves)[MOVE_NUM]) {
    int64_t t1, t2;
    types_ids(m.types, 2, t1, t2);
    const auto& sp = SPECIES(m.species);
    ids[0] = m.species;
    ids[1] = t1;
    ids[2] = t2;
    ids[3] = m.item;
    ids[4] = m.ability;
    for (int i = 0; i < 4; i++) {
        int mv = m.moves[i];
        ids[5 + i] = mv;
        ids[9 + i] = mv ? MOVE(mv).effect : EFFECT_NONE;
        ids[13 + i] = mv ? MOVE(mv).type : TYPE_NONE;
    }
    ids[17] = sp.abilities[0];
    ids[18] = sp.abilities[1];
    Writer w{num};
    w.put(1.0);
    w.put(0.0);
    w.put(is_active ? 1.0 : 0.0);
    w.put(1.0);
    w.put(m.hp == 0 ? 1.0 : 0.0);
    w.put(1.0);
    w.put(m.max_hp ? (double)m.hp / m.max_hp : 0.0);
    w.put(m.max_hp / 500.0);
    for (int k = 0; k < 5; k++) *w.p++ = (float)m.stats[k] / 500.0f;            // float32 division (numpy)
    for (int k = 0; k < 6; k++) *w.p++ = (float)sp.base[k] / 255.0f;
    w.onehot(m.status, 7);
    w.put(std::min(m.sleep_turns, 5) / 5.0);
    w.put(0.0);                                                                   // slot0 (swap only)
    w.put(m.level / 100.0);
    active_feats(w, active, m.max_hp);
    w.put(threat[0]);
    w.put(threat[1]);
    w.put(speed[0]);
    w.put(speed[1]);
    if (g_encode_version >= 4)
        for (int k = 0; k < N_DEFEATED; k++) w.put(0.0);                         // defeated-opponent record (swap only)
    for (int i = 0; i < 4; i++) {
        bool flags[4] = {false, false, false, false};
        int mv = m.moves[i];
        if (active && mv) {
            flags[0] = mv == active->encored_move;
            flags[1] = mv == active->disabled_move;
            flags[2] = mv == active->locked_move;
            flags[3] = mv == active->charging_move;
        }
        move_num(moves[i], mv, true, m.pp[i], true, usable ? usable[i] : false, flags, last_move && mv == last_move,
                 est[i]);
    }
}

void seen_mon(const ObsSeenMon& m, const ObsActive* active, bool is_active, int last_move, const double (*est)[3],
              int n_est, int64_t* ids, float* num, float (*moves)[MOVE_NUM]) {
    if (!m.seen) {
        ids[0] = SPECIES_UNK;
        ids[1] = ids[2] = TYPE_UNK;
        ids[3] = ITEM_UNK;
        ids[4] = ABILITY_UNK;
        for (int i = 0; i < 4; i++) {
            ids[5 + i] = MOVE_UNK;
            ids[9 + i] = EFFECT_UNK;
            ids[13 + i] = TYPE_UNK;
        }
        ids[17] = ids[18] = ABILITY_UNK;
        for (int k = 0, n = encode_mon_num(g_encode_version); k < n; k++) num[k] = 0.0f;
        num[1] = 1.0f;
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < MOVE_NUM; k++) moves[i][k] = 0.0f;
        return;
    }
    int64_t t1, t2;
    types_ids(m.types, m.n_types, t1, t2);
    int nrev = std::min(m.n_moves, 4);
    ids[0] = m.species;
    ids[1] = t1;
    ids[2] = t2;
    ids[3] = m.item >= 0 ? m.item : ITEM_UNK;
    ids[4] = m.revealed_ability ? m.revealed_ability : ABILITY_UNK;
    for (int i = 0; i < 4; i++) {
        if (i < nrev) {
            int mv = m.moves[i];
            ids[5 + i] = mv;
            ids[9 + i] = mv ? MOVE(mv).effect : EFFECT_NONE;
            ids[13 + i] = mv ? MOVE(mv).type : TYPE_NONE;
        } else {
            ids[5 + i] = MOVE_UNK;
            ids[9 + i] = EFFECT_UNK;
            ids[13 + i] = TYPE_UNK;
        }
    }
    ids[17] = m.n_abilities > 0 ? m.possible_abilities[0] : 0;
    ids[18] = m.n_abilities > 1 ? m.possible_abilities[1] : 0;
    Writer w{num};
    w.put(0.0);
    w.put(1.0);
    w.put(is_active ? 1.0 : 0.0);
    w.put(1.0);
    w.put(m.fainted ? 1.0 : 0.0);
    w.put(0.0);
    w.put(std::max(m.hp_pixels, 0) / 48.0);
    w.put(0.0);
    for (int k = 0; k < 5; k++) w.put(0.0);
    for (int k = 0; k < 6; k++) *w.p++ = (m.n_base == 6 ? (float)m.base_stats[k] : 0.0f) / 255.0f;
    w.onehot(m.status, 7);
    w.put(std::min(m.sleep_turns, 5) / 5.0);
    w.put(0.0);
    w.put(m.level / 100.0);
    active_feats(w, active, 0);
    for (int k = 0; k < 4; k++) w.put(0.0);
    if (g_encode_version >= 4)
        for (int k = 0; k < N_DEFEATED; k++) w.put(0.0);                         // defeated-opponent record (swap only)
    for (int i = 0; i < 4; i++) {
        bool known = i < nrev;
        int mvid = known ? m.moves[i] : 0;
        bool flags[4] = {false, false, false, false};
        if (active && known) {
            flags[0] = mvid == active->encored_move;
            flags[1] = mvid == active->disabled_move;
            flags[2] = mvid == active->locked_move;
            flags[3] = mvid == active->charging_move;
        }
        move_num(moves[i], mvid, known, 0, false, false, flags, known && last_move && mvid == last_move,
                 i < n_est ? est[i] : nullptr);
    }
}

void side_feats(Writer& w, const ObsSide& s) {
    w.put(s.reflect_turns / 5.0);
    w.put(s.light_screen_turns / 5.0);
    w.put(s.safeguard_turns / 5.0);
    w.put(s.mist_turns / 5.0);
    w.put(s.spikes / 3.0);
    w.put(s.future_sight_turns / 2.0);
    w.put(s.future_sight_move != 0 ? 1.0 : 0.0);
    w.put(s.wish_turns / 2.0);
}

}  // namespace

// ---- public API -----------------------------------------------------------------------------------------------

void obs_init(ObsMemory& mem, uint16_t hint_type, uint16_t hint_style) {
    std::memset(&mem, 0, sizeof(mem));
    mem.hint_type = hint_type;
    mem.hint_style = hint_style;
    for (int i = 0; i < 3; i++) {
        mem.revealed_items[i] = -1;
        mem.last_enemy_item[i] = -1;
        mem.bench_status[i] = -1;
    }
    mem.events = Observer::default_events();
}

void obs_observe(ObsMemory& mem, Gen3Game& game, bool forced, uint8_t unusable_mask, bool can_switch) {
    Observer{mem}.observe(game, forced, unusable_mask, can_switch);
}

void obs_rebase(ObsMemory& mem, Gen3Game& game, bool forced) { Observer{mem}.rebase(game, forced); }

void obs_finish(ObsMemory& mem, Gen3Game& game) { Observer{mem}.finish(game); }

void set_encode_version(int version) {
    if (version != 3 && version != 4)
        throw std::invalid_argument("the C++ encoder supports encoding versions 3 and 4");
    g_encode_version = version;
}

int encode_version() { return g_encode_version; }

int encode_mon_num(int version) { return version >= 4 ? MON_NUM_V4 : MON_NUM_V3; }

void encode_view(const ObsView& v, const EncodeCtx& ctx, EncodedObs& out) {
    const int version = g_encode_version, mon_w = encode_mon_num(version);
    out.mon_w = mon_w;
    out.move_w = MOVE_NUM;
    out.ctx_w = CTX_NUM;
    auto moves_of = [&out](int i) { return reinterpret_cast<float (*)[MOVE_NUM]>(out.move_num + i * 4 * MOVE_NUM); };
    Estimates es;
    battle_estimates(v, ctx, version, es);
    const ObsEvents& lt = v.last_turn;
    int a_own = v.own_active.party_index, a_foe = v.enemy_active.party_index;
    for (int i = 0; i < 3; i++) {
        bool act = i == a_own;
        own_mon(v.own[i], act ? &v.own_active : nullptr, act, act ? v.usable : nullptr, act ? lt.own_move : 0,
                es.own[i], es.threat[i], es.speed[i], out.mon_ids[i], out.mon_num + i * mon_w, moves_of(i));
    }
    for (int i = 0; i < 3; i++) {
        bool act = i == a_foe;
        seen_mon(v.enemy[i], act ? &v.enemy_active : nullptr, act, act ? lt.enemy_move : 0, es.foe,
                 act ? es.n_foe : 0, out.mon_ids[3 + i], out.mon_num + (3 + i) * mon_w, moves_of(3 + i));
    }
    int own_max = v.own[std::max(0, std::min(2, a_own))].max_hp;
    if (!own_max) own_max = 1;
    Writer w{out.ctx_num};
    w.onehot(v.weather, 5);
    w.put(v.weather_permanent ? 1.0 : 0.0);
    w.put(v.weather_turns_left / 5.0);
    side_feats(w, v.own_side);
    side_feats(w, v.enemy_side);
    w.put(std::min(v.turn, 50) / 50.0);
    w.put(v.forced_switch ? 1.0 : 0.0);
    w.put(v.must_struggle ? 1.0 : 0.0);
    w.onehot(lt.first + 1, 3);
    w.onehot(lt.own_action, 4);
    w.onehot(lt.own_crit + 1, 3);
    w.onehot(lt.enemy_action, 4);
    w.onehot(lt.enemy_crit + 1, 3);
    w.put(lt.damage_dealt_pixels / 48.0);
    w.put(std::min((double)lt.damage_taken / own_max, 1.0));
    w.put(lt.own_fainted ? 1.0 : 0.0);
    w.put(lt.enemy_fainted ? 1.0 : 0.0);
    w.put(lt.in_progress ? 1.0 : 0.0);
    w.put(std::min(lt.turns, 2) / 2.0);
    w.onehot(v.hint_type, N_HINT_TYPES);
    w.onehot(v.hint_style, N_STYLES);
    // _context
    int rank = 0;
    for (int t : RENT_RANKS) rank += ctx.rents >= t;
    w.put(std::log1p((double)ctx.streak) / std::log1p(100.0));
    w.onehot(ctx.battle, 7);
    w.onehot(std::min(ctx.challenge, N_ROUNDS - 1), N_ROUNDS);
    w.onehot(rank, 6);
    w.put(std::min(ctx.rents, 50) / 50.0);
    w.put((ctx.streak % 7 == 0 && ctx.streak > 0) ? 1.0 : 0.0);
    out.ctx_ids[0] = lt.own_move;
    out.ctx_ids[1] = lt.enemy_move;
    for (int i = 0; i < 7; i++) out.mask[i] = v.mask[i];
    out.active = a_own;
}

void encode_battle(const ObsMemory& mem, Gen3Game&, const EncodeCtx& ctx, EncodedObs& out) {
    encode_view(mem.view, ctx, out);
}

}  // namespace pkmn
