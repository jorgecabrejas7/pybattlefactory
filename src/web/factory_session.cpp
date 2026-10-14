#include "factory_session.hpp"

extern "C" {
#include "gen3/observer_host.h"
}

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace pkmn::web {

namespace {

// ---- constants (pybattle/backend.py, pybattle/emu/decode.py, pybattle/view.py) ----
constexpr uint16_t FLAG_SYS_FACTORY_SILVER = 0x860 + 0x6C, FLAG_SYS_FACTORY_GOLD = 0x860 + 0x6D;
constexpr uint32_t G_PLAYER_PARTY = 0x20244ec, G_ENEMY_PARTY = 0x2024744, G_BATTLE_MONS = 0x2024084;
constexpr uint32_t SB2_FRONTIER_CUR_BATTLE_NUM = 0xCB2, SB2_FACTORY_RENTS_COUNT = 0xDF2;
constexpr int POKEMON_SIZE = 100, HP_BAR_PIXELS = 48;

int symbolsForStreak(int streak) { return int(streak >= 21) + int(streak >= 42); }

int factoryBrainStatus(int streak, int symbols) {
    int s = streak + 1;
    if (symbols < 2) return s == (symbols == 0 ? 21 : 42) ? symbols + 1 : 0;
    if (s == 21) return 3;
    return (s == 42 || (s > 42 && (s - 42) % 21 == 0)) ? 4 : 0;
}

int majorStatus(uint32_t status1) {
    if (status1 & 0x7) return 1;
    if (status1 & (1u << 7)) return 6;
    if (status1 & (1u << 3)) return 2;
    if (status1 & (1u << 4)) return 3;
    if (status1 & (1u << 5)) return 4;
    if (status1 & (1u << 6)) return 5;
    return 0;
}

const Gen3ObsSpecies& species(int sp) {
    if (sp < 0 || sp >= GEN3_OBS_NUM_SPECIES) throw std::out_of_range("species");
    return Gen3Obs_Tables()->species[sp];
}

int abilityOf(int sp, int abilityNum) {
    const auto& a = species(sp).abilities;
    return (abilityNum && a[1]) ? a[1] : a[0];
}

uint32_t rd32(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | p[1] << 8); }

// pokemon substructure order by personality % 24 (G, A, E, M positions)
constexpr uint8_t SUBSTRUCT_ORDER[24][4] = {
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 3, 1, 2}, {0, 2, 3, 1}, {0, 3, 2, 1},
    {1, 0, 2, 3}, {1, 0, 3, 2}, {2, 0, 1, 3}, {3, 0, 1, 2}, {2, 0, 3, 1}, {3, 0, 2, 1},
    {1, 2, 0, 3}, {1, 3, 0, 2}, {2, 1, 0, 3}, {3, 1, 0, 2}, {2, 3, 0, 1}, {3, 2, 0, 1},
    {1, 2, 3, 0}, {1, 3, 2, 0}, {2, 1, 3, 0}, {3, 1, 2, 0}, {2, 3, 1, 0}, {3, 2, 1, 0}};

// ---- a small JSON writer ----
class Json {
public:
    std::string out;
    Json& beginObj() { sep(); out += '{'; m_first = true; return *this; }
    Json& endObj() { out += '}'; m_first = false; return *this; }
    Json& beginArr() { sep(); out += '['; m_first = true; return *this; }
    Json& endArr() { out += ']'; m_first = false; return *this; }
    Json& key(const char* k) { sep(); out += '"'; out += k; out += "\":"; m_first = true; return *this; }
    Json& num(long long v) { sep(); out += std::to_string(v); return *this; }
    Json& boolean(bool v) { sep(); out += v ? "true" : "false"; return *this; }
    Json& null() { sep(); out += "null"; return *this; }
    Json& str(const char* s) { sep(); out += '"'; out += s; out += '"'; return *this; }
    Json& strEsc(const std::string& s) {
        sep();
        out += '"';
        for (unsigned char c : s) {
            if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
            else if (c == '\n') out += "\\n";
            else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
            else out += char(c);
        }
        out += '"';
        return *this;
    }
    template <class T> Json& arr(const T* v, int n) { beginArr(); for (int i = 0; i < n; i++) num(v[i]); return endArr(); }
    Json& real(double v) { sep(); char b[32]; std::snprintf(b, sizeof b, "%.6g", v); out += b; return *this; }
    Json& kv(const char* k, long long v) { return key(k).num(v); }
    Json& kb(const char* k, bool v) { return key(k).boolean(v); }
    template <class T> Json& ka(const char* k, const T* v, int n) { return key(k).arr(v, n); }

private:
    void sep() { if (!m_first) out += ','; m_first = false; }
    bool m_first = true;
};

const char* const ACTIVE_BOOLS[26] = {"confused", "infatuated", "substitute", "leech_seeded", "cursed", "nightmare",
                                      "trapped", "focus_energy", "transformed", "perish_song", "rooted", "yawn",
                                      "torment", "taunted", "must_recharge", "destiny_bond", "defense_curl",
                                      "foresight", "minimized", "charged_up", "imprisoning", "grudge", "mud_sport",
                                      "water_sport", "rage", "first_turn"};
const char* const ACTIVE_COUNTERS[16] = {"confusion_turns", "toxic_counter", "taunt_turns_left", "encore_turns",
                                         "disable_turns", "yawn_turns_left", "wrapped_turns", "uproar_turns",
                                         "rampage_turns", "bide_turns_left", "rollout_hits_left", "fury_cutter_count",
                                         "stockpile", "charge_turns_left", "locked_on_turns_left", "protect_uses"};

// OwnMon.from_party (+ the observer's sleep_turns in battle)
void ownMon(Json& j, const PartyMon& m, int sleepTurns) {
    const auto& sp = species(m.species);
    int types[2] = {sp.types[0], sp.types[1]};
    j.beginObj().kv("species", m.species).kv("level", m.level).kv("hp", m.hp).kv("max_hp", m.max_hp)
        .ka("stats", m.stats, 5).ka("moves", m.moves, 4).ka("pp", m.pp, 4).kv("item", m.item)
        .kv("ability", abilityOf(m.species, m.ability_num)).kv("nature", m.nature()).ka("ivs", m.ivs, 6)
        .ka("evs", m.evs, 6).kv("status", majorStatus(m.status)).ka("types", types, 2).kv("sleep_turns", sleepTurns)
        .endObj();
}

// SeenMon from species knowledge + what was seen
void seenMon(Json& j, const ObsSeenMon& m) {
    j.beginObj().kv("species", m.species).kv("level", m.level).kv("hp_pixels", m.hp_pixels).kv("status", m.status)
        .ka("revealed_moves", m.moves, m.n_moves).key("revealed_item");
    if (m.item < 0) j.null(); else j.num(m.item);
    j.kb("fainted", m.fainted).kb("seen", m.seen).ka("types", m.types, m.n_types)
        .ka("base_stats", m.base_stats, m.n_base).ka("possible_abilities", m.possible_abilities, m.n_abilities)
        .kv("revealed_ability", m.revealed_ability).kv("sleep_turns", m.sleep_turns).endObj();
}

void activeState(Json& j, const ObsActive& a) {
    j.beginObj().kv("party_index", a.party_index).ka("stat_stages", a.stat_stages, 7).ka("types", a.types, 2);
    for (int k = 0; k < 26; k++) j.kb(ACTIVE_BOOLS[k], a.bools[k]);
    for (int k = 0; k < 16; k++) j.kv(ACTIVE_COUNTERS[k], a.counters[k]);
    j.kv("perish_count", a.perish_count).kv("substitute_hp", a.substitute_hp)
        .kv("semi_invulnerable", a.semi_invulnerable).kv("encored_move", a.encored_move)
        .kv("disabled_move", a.disabled_move).kv("locked_move", a.locked_move).kv("charging_move", a.charging_move)
        .endObj();
}

void sideState(Json& j, const ObsSide& s) {
    j.beginObj().kv("reflect_turns", s.reflect_turns).kv("light_screen_turns", s.light_screen_turns)
        .kv("safeguard_turns", s.safeguard_turns).kv("mist_turns", s.mist_turns).kv("spikes", s.spikes)
        .kv("future_sight_turns", s.future_sight_turns).kv("future_sight_move", s.future_sight_move)
        .kv("wish_turns", s.wish_turns).endObj();
}

void turnEvents(Json& j, const ObsEvents& e) {
    j.beginObj().kv("turn", e.turn).kv("turns", e.turns).kb("in_progress", e.in_progress).kv("first", e.first)
        .kv("own_action", e.own_action).kv("own_move", e.own_move).kv("own_crit", e.own_crit)
        .kv("enemy_action", e.enemy_action).kv("enemy_move", e.enemy_move).kv("enemy_crit", e.enemy_crit)
        .kv("damage_dealt_pixels", e.damage_dealt_pixels).kv("damage_taken", e.damage_taken)
        .kb("own_fainted", e.own_fainted).kb("enemy_fainted", e.enemy_fainted).endObj();
}

// GetExtCtrlCodeLength (string_util.c): argument bytes after EXT_CTRL_CODE_BEGIN + code
constexpr uint8_t EXT_CTRL_ARGS[0x18] = {0, 1, 1, 1, 3, 1, 1, 0, 1, 0, 0, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0, 0, 0};

void putUtf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += char(cp);
    else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
    else { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
}

// game charmap (charmap.txt) for the bytes the game's own code writes (digits, letters, punctuation)
uint32_t charmap(uint8_t c) {
    if (c >= 0xA1 && c <= 0xAA) return '0' + (c - 0xA1);
    if (c >= 0xBB && c <= 0xD4) return 'A' + (c - 0xBB);
    if (c >= 0xD5 && c <= 0xEE) return 'a' + (c - 0xD5);
    switch (c) {
    case 0x00: return ' ';
    case 0xAB: return '!';
    case 0xAC: return '?';
    case 0xAD: return '.';
    case 0xAE: return '-';
    case 0xB0: return 0x2026;   // …
    case 0xB1: return 0x201C;
    case 0xB2: return 0x201D;
    case 0xB3: return 0x2018;
    case 0xB4: return 0x2019;
    case 0xB5: return 0x2642;   // ♂
    case 0xB6: return 0x2640;   // ♀
    case 0xB8: return ',';
    case 0xBA: return '/';
    case 0xF0: return ':';
    default: return 0;
    }
}

}  // namespace

std::string game_text_utf8(const std::string& text) {
    std::string out;
    const auto* p = reinterpret_cast<const unsigned char*>(text.data());
    size_t n = text.size();
    for (size_t i = 0; i < n; i++) {
        unsigned char c = p[i];
        if (c == 0xFF) break;
        if (c == 0xFE || c == 0xFA || c == 0xFB) {   // newline, scroll, prompt
            if (!out.empty() && out.back() != '\n') out += '\n';
            continue;
        }
        if (c == 0xFC) {                              // EXT_CTRL_CODE_BEGIN code args...
            if (i + 1 < n && p[i + 1] < sizeof EXT_CTRL_ARGS) i += 1 + EXT_CTRL_ARGS[p[i + 1]];
            else i += 1;
            continue;
        }
        if (c == 0xFD) { i++; continue; }            // an unexpanded placeholder
        if (c >= 0x20 && c < 0x7F) { out += char(c); continue; }
        // UTF-8 sequences from the host build's literals
        int len = (c >= 0xC2 && c <= 0xDF) ? 2 : (c >= 0xE0 && c <= 0xEF) ? 3 : (c >= 0xF0 && c <= 0xF4) ? 4 : 0;
        bool utf8 = len && i + len <= n;
        for (int k = 1; utf8 && k < len; k++) utf8 = (p[i + k] & 0xC0) == 0x80;
        if (utf8) { out.append(reinterpret_cast<const char*>(p + i), size_t(len)); i += len - 1; continue; }
        if (uint32_t cp = charmap(c)) putUtf8(out, cp);
    }
    // the host's literals keep the decomp's {FEMALE} / {MALE} spellings
    for (auto [from, to] : {std::pair<const char*, const char*>{"{FEMALE}", "\u2640"}, {"{MALE}", "\u2642"}}) {
        for (size_t at; (at = out.find(from)) != std::string::npos;) out.replace(at, std::strlen(from), to);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out;
}

PartyMon decode_pokemon(const uint8_t* data) {
    PartyMon m{};
    m.personality = rd32(data);
    uint32_t otId = rd32(data + 4);
    uint32_t key = m.personality ^ otId;
    uint8_t secure[48];
    for (int w = 0; w < 12; w++) {
        uint32_t v = rd32(data + 0x20 + 4 * w) ^ key;
        for (int b = 0; b < 4; b++) secure[4 * w + b] = uint8_t(v >> (8 * b));
    }
    const uint8_t* order = SUBSTRUCT_ORDER[m.personality % 24];
    const uint8_t* g = secure + order[0] * 12;
    const uint8_t* a = secure + order[1] * 12;
    const uint8_t* e = secure + order[2] * 12;
    const uint8_t* mi = secure + order[3] * 12;
    m.species = rd16(g);
    m.item = rd16(g + 2);
    for (int i = 0; i < 4; i++) m.moves[i] = rd16(a + 2 * i), m.pp[i] = a[8 + i];
    for (int i = 0; i < 6; i++) m.evs[i] = e[i];
    uint32_t iv = rd32(mi + 4);
    for (int i = 0; i < 6; i++) m.ivs[i] = int((iv >> (5 * i)) & 31);
    m.ability_num = int((iv >> 31) & 1);
    m.status = rd32(data + 0x50);
    m.level = data[0x54];
    m.hp = rd16(data + 0x56);
    m.max_hp = rd16(data + 0x58);
    for (int i = 0; i < 5; i++) m.stats[i] = rd16(data + 0x5A + 2 * i);
    return m;
}

FactorySession::FactorySession(bool openLevel, int maxTurns) : m_openLevel(openLevel), m_maxTurns(maxTurns) {
    m_game.setMessageLog(true);
    m_game.setMessageText(true);
    obs_init(m_obs);
    obs_init(m_obsDone);
    std::memset(&m_enc, 0, sizeof m_enc);
}

void FactorySession::reset(uint32_t seed, int winStreak, int rentsCount) {
    m_game = Gen3Game();
    int symbols = symbolsForStreak(winStreak);
    m_game.setFlag(FLAG_SYS_FACTORY_SILVER, symbols >= 1);
    m_game.setFlag(FLAG_SYS_FACTORY_GOLD, symbols >= 2);
    m_game.factoryBegin(m_openLevel, uint16_t(winStreak), uint16_t(rentsCount), seed);
    m_game.takeMessages();
    m_startStreak = winStreak;
    m_turns = 0;
    obs_init(m_obs);
    m_hasDone = false;
    m_lastWon = -1;
    sync();
}

int FactorySession::wins() { return m_game.factoryInfo().wins; }

void FactorySession::sync() {
    for (;;) {
        int fp = m_game.factoryPhase();
        if (fp == Gen3Game::F_RENTAL) { m_phase = Phase::RENTAL; return; }
        if (fp == Gen3Game::F_SWAP) { m_phase = Phase::SWAP; return; }
        if (fp == Gen3Game::F_RUN_OVER) { m_phase = Phase::RUN_OVER; m_lastWon = 0; return; }
        // the battle first runs to its end (gBattleOutcome set) so the observer sees the last turn; then
        // factoryRunBattle winds it down (SimBackend._sync)
        int d = m_game.run(400000);
        if (d == Gen3Game::BATTLE_OVER) {
            obs_finish(m_obs, m_game);
            d = m_game.factoryRunBattle();
        }
        if (d == Gen3Game::BATTLE_OVER) {
            m_lastWon = m_game.factoryInfo().lastOutcome == 1 ? 1 : 0;
            m_obsDone = m_obs;
            m_hasDone = true;
            if (m_game.factoryPhase() != Gen3Game::F_SWAP) obs_init(m_obs);
            m_turns = 0;
            continue;
        }
        if (d == Gen3Game::TIMEOUT) throw std::runtime_error("gen3 battle did not reach a decision");
        m_phase = d == Gen3Game::SWITCH ? Phase::FORCED_SWITCH : Phase::BATTLE;
        // BattleObserver.observe at every decision, in order (SimBackend.view / act)
        obs_observe(m_obs, m_game, m_phase == Phase::FORCED_SWITCH, m_game.unusableMoves(0), m_game.canSwitch(0));
        return;
    }
}

void FactorySession::newObserver() {
    auto fi = m_game.factoryInfo();
    obs_init(m_obs, uint16_t(fi.hintType), uint16_t(fi.hintStyle));
}

FactorySession::Info FactorySession::runInfo() {
    if (m_mirror) return Info{m_ctx.streak, m_ctx.battle, m_ctx.challenge, m_ctx.rents, m_ctx.noland};
    auto fi = m_game.factoryInfo();
    Info r{};
    r.streak = m_startStreak + fi.wins;
    std::string bn = m_game.readSaveBlock2(SB2_FRONTIER_CUR_BATTLE_NUM, 2);
    std::string rc = m_game.readSaveBlock2(SB2_FACTORY_RENTS_COUNT + 2 * int(m_openLevel), 2);
    r.battle = rd16(reinterpret_cast<const uint8_t*>(bn.data()));
    r.rents = rd16(reinterpret_cast<const uint8_t*>(rc.data()));
    r.challenge = r.streak / 7;
    if (m_phase == Phase::BATTLE || m_phase == Phase::FORCED_SWITCH) r.noland = fi.brainStatus != 0;
    else r.noland = factoryBrainStatus(r.streak, symbolsForStreak(r.streak)) != 0;
    return r;
}

std::string FactorySession::runInfoJson() {
    Info r = runInfo();
    Json j;
    j.beginObj().kv("win_streak", r.streak).kv("battle_in_challenge", r.battle).kv("challenge_num", r.challenge)
        .kb("open_level", m_openLevel).kv("wins", wins()).kv("rents", r.rents).kb("noland", r.noland)
        // beyond RunInfo: the opponent trainer (Battle Frontier trainer id; shown before the battle)
        .kv("trainer_id", m_game.factoryInfo().trainerId).endObj();
    return j.out;
}

std::string FactorySession::swapJson(const std::string& own, const std::string& foe, int hintType, int hintStyle) {
    Json j;
    j.beginObj().key("own_party").beginArr();
    for (int i = 0; i < 3; i++)
        ownMon(j, decode_pokemon(reinterpret_cast<const uint8_t*>(own.data()) + i * POKEMON_SIZE), 0);
    j.endArr().key("enemy_party").beginArr();
    for (int i = 0; i < 3; i++) {
        // BattleObserver.swap_candidates: the swap screen's species + what the battle revealed
        PartyMon m = decode_pokemon(reinterpret_cast<const uint8_t*>(foe.data()) + i * POKEMON_SIZE);
        const auto& sp = species(m.species);
        ObsSeenMon s{};
        s.seen = true;
        s.species = m.species;
        s.level = m.level;
        s.hp_pixels = HP_BAR_PIXELS;
        s.n_moves = m_obs.n_revealed[i];
        for (int k = 0; k < s.n_moves; k++) s.moves[k] = m_obs.revealed_moves[i][k];
        s.item = m_obs.revealed_items[i];
        s.n_types = 2, s.types[0] = sp.types[0], s.types[1] = sp.types[1];
        s.n_base = 6;
        for (int k = 0; k < 6; k++) s.base_stats[k] = sp.base[k];
        s.n_abilities = 1, s.possible_abilities[0] = sp.abilities[0];
        if (sp.abilities[1] && sp.abilities[1] != sp.abilities[0])
            s.possible_abilities[s.n_abilities++] = sp.abilities[1];
        s.revealed_ability = m_obs.revealed_abilities[i];
        seenMon(j, s);
    }
    j.endArr().kv("hint_type", hintType).kv("hint_style", hintStyle).key("defeated").beginArr();
    for (int i = 0; i < 3; i++) {
        const int32_t* r = m_obs.records[i];
        j.beginObj().kv("damage", r[0]).kv("team_max_hp", m_obs.team_max_hp).kv("knockouts", r[1])
            .kv("turns", r[2]).kv("hits_taken", r[3]).kv("max_boosts", r[4]).kb("inflicted_status", r[5] != 0)
            .endObj();
    }
    j.endArr().endObj();
    return j.out;
}

std::string FactorySession::swapViewJson(const std::string& own, const std::string& enemy, int hintType, int hintStyle) {
    return swapJson(own, enemy, hintType, hintStyle);
}

std::string FactorySession::viewJson() {
    Json j;
    auto fi = m_game.factoryInfo();
    switch (m_phase) {
    case Phase::RENTAL: {
        j.beginObj().key("candidates").beginArr();
        for (int i = 0; i < 6; i++) {
            std::string raw = m_game.factoryRental(i);
            ownMon(j, decode_pokemon(reinterpret_cast<const uint8_t*>(raw.data())), 0);
        }
        j.endArr().key("frontier_ids").beginArr();
        for (int i = 0; i < 6; i++) j.num(m_game.factoryRentalMonId(i));
        j.endArr().kv("hint_type", fi.hintType).kv("hint_style", fi.hintStyle).endObj();
        return j.out;
    }
    case Phase::SWAP:
        return swapJson(m_game.readRam(G_PLAYER_PARTY, 3 * POKEMON_SIZE), m_game.readRam(G_ENEMY_PARTY, 3 * POKEMON_SIZE),
                        fi.hintType, fi.hintStyle);
    case Phase::BATTLE:
    case Phase::FORCED_SWITCH: {
        const ObsView& v = m_obs.view;
        std::string own = m_game.readRam(G_PLAYER_PARTY, 3 * POKEMON_SIZE);
        j.beginObj().key("own_party").beginArr();
        for (int i = 0; i < 3; i++) {
            PartyMon m = decode_pokemon(reinterpret_cast<const uint8_t*>(own.data()) + i * POKEMON_SIZE);
            ownMon(j, m, v.own[i].sleep_turns);
        }
        j.endArr().key("own_active");
        activeState(j, v.own_active);
        j.key("enemy_party").beginArr();
        for (int i = 0; i < 3; i++) seenMon(j, v.enemy[i]);
        j.endArr().key("enemy_active");
        activeState(j, v.enemy_active);
        j.kv("weather", v.weather).kb("weather_permanent", v.weather_permanent)
            .kv("weather_turns_left", v.weather_turns_left).key("own_side");
        sideState(j, v.own_side);
        j.key("enemy_side");
        sideState(j, v.enemy_side);
        j.kv("turn", v.turn).kb("forced_switch", v.forced_switch).key("usable_moves").beginArr();
        for (int i = 0; i < 4; i++) j.boolean(v.usable[i]);
        j.endArr().key("switch_targets").beginArr();
        for (int i = 0; i < 3; i++) if (v.mask[4 + i]) j.num(i);
        j.endArr().key("last_turn");
        turnEvents(j, v.last_turn);
        j.kv("hint_type", v.hint_type).kv("hint_style", v.hint_style).kb("must_struggle", v.must_struggle)
            .key("legal_actions").beginArr();
        for (int i = 0; i < 4; i++) if (v.mask[i]) j.beginArr().str("move").num(i).endArr();
        for (int i = 0; i < 3; i++) if (v.mask[4 + i]) j.beginArr().str("switch").num(i).endArr();
        if (!v.forced_switch) j.beginArr().str("forfeit").endArr();
        j.endArr().endObj();
        return j.out;
    }
    case Phase::RUN_OVER:
        break;
    }
    return "null";
}

std::string FactorySession::takeLogJson() {
    Json j;
    j.beginArr();
    for (const auto& m : m_game.takeMessages()) {
        auto hpPixels = [](int hp, int maxHp) {
            if (maxHp <= 0) return 0;
            int v = hp * HP_BAR_PIXELS / maxHp;
            return v == 0 && hp > 0 ? 1 : v;
        };
        j.beginObj().kv("id", m.stringId).key("text").strEsc(game_text_utf8(m.text)).kv("attacker", m.attacker)
            .kv("target", m.target).kv("effect", m.effect).kv("move", m.move)
            .key("own").beginObj().kv("species", m.species[0]).kv("hp", m.hp[0]).kv("max_hp", m.maxHp[0])
            .kv("status", majorStatus(m.status1[0])).endObj()
            .key("foe").beginObj().kv("species", m.species[1]).kv("hp_pixels", hpPixels(m.hp[1], m.maxHp[1]))
            .kv("status", majorStatus(m.status1[1])).endObj()
            .endObj();
    }
    j.endArr();
    return j.out;
}

void FactorySession::actRental(int a, int b, int c) {
    if (m_phase != Phase::RENTAL) throw std::runtime_error("not at the rental screen");
    for (int x : {a, b, c}) if (x < 0 || x > 5) throw std::out_of_range("rental slot must be 0-5");
    ObsMemory saved = m_obs;
    newObserver();
    if (!m_game.factoryRent(a, b, c)) {
        m_obs = saved;
        throw std::invalid_argument("rental refused (same species twice?)");
    }
    sync();
}

void FactorySession::actSwap(int ownSlot, int enemySlot) {
    if (m_phase != Phase::SWAP) throw std::runtime_error("not at the swap screen");
    bool ok = ownSlot < 0 ? m_game.factorySwap(-1, 0) : m_game.factorySwap(ownSlot, enemySlot);
    if (!ok) throw std::invalid_argument("swap refused (species already on the team)");
    newObserver();
    sync();
}

void FactorySession::beforeBattleAction() {
    if (m_phase != Phase::BATTLE) throw std::runtime_error("not at a battle decision");
}

void FactorySession::actMove(int slot) {
    beforeBattleAction();
    if (m_turns + 1 > m_maxTurns) return actForfeit();
    m_game.chooseMove(slot);    // throws (nothing changed) when refused
    m_turns++;
    sync();
}

void FactorySession::actSwitch(int partyIndex) {
    if (m_phase == Phase::FORCED_SWITCH) {
        m_game.chooseSwitch(partyIndex);
        sync();
        return;
    }
    beforeBattleAction();
    if (m_turns + 1 > m_maxTurns) return actForfeit();
    m_game.chooseSwitch(partyIndex);
    m_turns++;
    sync();
}

void FactorySession::actForfeit() {
    beforeBattleAction();
    m_turns++;
    m_game.forfeit();
    sync();
}

const EncodedObs& FactorySession::encodeBattle() {
    if (m_phase != Phase::BATTLE && m_phase != Phase::FORCED_SWITCH) throw std::runtime_error("not in battle");
    Info r = runInfo();
    std::memset(&m_enc, 0, sizeof m_enc);
    // the v4 layout (= v5's battle encoding) whatever the process-wide version is
    int version = encode_version();
    set_encode_version(4);
    encode_battle(m_obs, m_game, EncodeCtx{r.streak, r.battle, r.challenge, r.rents}, m_enc);
    set_encode_version(version);
    return m_enc;
}

std::string FactorySession::estimatesJson() {
    if (m_phase != Phase::BATTLE && m_phase != Phase::FORCED_SWITCH) return "null";
    Info r = runInfo();
    BattleEstimates es;
    estimate_battle(m_obs.view, EncodeCtx{r.streak, r.battle, r.challenge, r.rents}, 4, es);
    const ObsView& v = m_obs.view;
    const ObsSeenMon& foe = v.enemy[std::max(0, std::min(2, v.enemy_active.party_index))];
    Json j;
    j.beginObj().key("own").beginArr();
    for (int i = 0; i < 3; i++) {
        j.beginArr();
        for (int k = 0; k < 4; k++) j.beginArr().real(es.own[i][k][0]).real(es.own[i][k][1]).real(es.own[i][k][2]).endArr();
        j.endArr();
    }
    j.endArr().key("foe").beginArr();
    for (int k = 0; k < es.n_foe && k < foe.n_moves; k++)
        j.beginObj().kv("move", foe.moves[k]).key("min").real(es.foe[k][0]).key("max").real(es.foe[k][1])
            .key("ko").real(es.foe[k][2]).endObj();
    j.endArr().key("threat").beginArr();
    for (int i = 0; i < 3; i++) j.beginArr().real(es.threat[i][0]).real(es.threat[i][1]).endArr();
    j.endArr().key("speed").beginArr();
    for (int i = 0; i < 3; i++) j.beginArr().real(es.speed[i][0]).real(es.speed[i][1]).endArr();
    j.endArr().endObj();
    return j.out;
}

// ---- mirror mode ----

namespace {
constexpr uint16_t VAR_FRONTIER_BATTLE_MODE = 0x40CE, VAR_FRONTIER_FACILITY = 0x40CF;
constexpr int FRONTIER_FACILITY_FACTORY = 4, FRONTIER_MODE_SINGLES = 0;
constexpr uint32_t SB2_FRONTIER_LVL_MODE = 0xCA9, SB2_FACTORY_WIN_STREAKS = 0xDE2, SB2_TOWER_WIN_STREAKS = 0xCE0,
                   SB2_RENTAL_MONS = 0xE70;
std::string le16s(const uint16_t* v, int n) {
    std::string out;
    for (int i = 0; i < n; i++) { out += char(v[i] & 0xFF); out += char(v[i] >> 8); }
    return out;
}
}  // namespace

void FactorySession::mirrorBegin(const BattleStart& st, int hintType, int hintStyle) {
    // pybattle/diff.py setup_battle
    m_mirror = true;
    m_mirrorSynced = false;
    m_atDecision = false;
    m_mirrorError.clear();
    m_game = Gen3Game();
    m_game.setBattle(st.battleTypeFlags, st.trainerId);
    m_game.writeParty(0, st.playerParty);
    m_game.writeParty(1, st.enemyParty);
    m_game.setVar(VAR_FRONTIER_FACILITY, FRONTIER_FACILITY_FACTORY);
    m_game.setVar(VAR_FRONTIER_BATTLE_MODE, FRONTIER_MODE_SINGLES);
    uint8_t lvl = uint8_t(m_game.readSaveBlock2(SB2_FRONTIER_LVL_MODE, 1)[0]);
    m_game.writeSaveBlock2(SB2_FRONTIER_LVL_MODE, std::string(1, char((lvl & ~3) | st.lvlMode)));
    m_game.writeSaveBlock2(SB2_FRONTIER_CUR_BATTLE_NUM, le16s(&st.battleNum, 1));
    m_game.writeSaveBlock2(SB2_FACTORY_WIN_STREAKS, le16s(st.factoryStreaks, 4));
    m_game.writeSaveBlock2(SB2_TOWER_WIN_STREAKS, le16s(st.towerStreaks, 8));
    if (!st.rentalMons.empty()) m_game.writeSaveBlock2(SB2_RENTAL_MONS, st.rentalMons);
    if (!m_game.start(st.rng)) {
        m_mirrorError = "the battle did not start";
        return;
    }
    m_game.setMessageLog(true);
    m_game.setMessageText(true);
    obs_init(m_obs, uint16_t(hintType), uint16_t(hintStyle));
    m_mirrorSynced = true;
}

// pybattle/diff.py _mask_battle_mons: bytes the game fills from uninitialised stack memory (BattlePokemon.unknown,
// the nickname / OT name bytes after their terminator) are not compared.
static std::string maskBattleMons(std::string b) {
    for (size_t base = 0; base + 0x58 <= b.size(); base += 0x58) {
        b[base + 0x23] = 0;
        for (auto [start, len] : {std::pair<size_t, size_t>{0x30, 11}, {0x3C, 8}}) {
            size_t end = len;
            for (size_t k = 0; k < len; k++)
                if (uint8_t(b[base + start + k]) == 0xFF) { end = k; break; }
            for (size_t k = end + 1; k < len; k++) b[base + start + k] = 0;
        }
    }
    return b;
}

bool FactorySession::matches(Gen3Game& g, const std::string& snap) {
    size_t at = 0;
    auto take = [&](size_t n) { std::string s = snap.substr(at, n); at += n; return s; };
    std::string rng = take(4);
    uint32_t want = uint32_t(uint8_t(rng[0])) | uint32_t(uint8_t(rng[1])) << 8 | uint32_t(uint8_t(rng[2])) << 16 |
                    uint32_t(uint8_t(rng[3])) << 24;
    if (g.rng() != want) return false;
    for (const Region& r : m_regions) {
        std::string want = take(r.size), got = g.readRam(r.addr, r.size);
        if (r.addr == G_BATTLE_MONS) { want = maskBattleMons(want); got = maskBattleMons(got); }
        if (got != want) return false;
    }
    if (g.readRam(G_PLAYER_PARTY, 300) != take(300)) return false;
    if (g.readRam(G_ENEMY_PARTY, 300) != take(300)) return false;
    return at == snap.size();
}

int FactorySession::mirrorSync(bool forced, const std::string& snap) {
    if (!m_mirror || !m_mirrorSynced) return 0;
    // nothing happened since the last decision (a menu opened and closed, the voluntary party menu of a switch)
    if (m_atDecision && matches(m_game, snap)) return 2;
    const int want = forced ? Gen3Game::SWITCH : Gen3Game::ACTION;
    bool found = false;
    if (!m_atDecision) {
        // the battle's first decision (or the first after a skipped one)
        found = m_game.run(400000) == want && matches(m_game, snap);
    } else {
        // the player acted at the last decision: find which action leads here
        const ObsView& v = m_obs.view;
        for (int k = 0; k < 7 && !found; k++) {
            if (!v.mask[k]) continue;
            Gen3Game c = m_game;
            try {
                if (k < 4) c.chooseMove(k); else c.chooseSwitch(k - 4);
            } catch (const std::exception&) { continue; }
            if (c.run(400000) == want && matches(c, snap)) {
                m_game = c;
                found = true;
            }
        }
    }
    if (!found) {
        m_mirrorSynced = false;
        m_mirrorError = "the simulator does not reproduce this decision";
        return 0;
    }
    m_game.setMessageLog(true);
    m_game.setMessageText(true);
    m_phase = forced ? Phase::FORCED_SWITCH : Phase::BATTLE;
    obs_observe(m_obs, m_game, forced, m_game.unusableMoves(0), m_game.canSwitch(0));
    m_atDecision = true;
    return 1;
}

bool FactorySession::mirrorFinish(const std::string& snap) {
    if (!m_mirror || !m_mirrorSynced || !m_atDecision) return false;
    const ObsView& v = m_obs.view;
    for (int k = -1; k < 7; k++) {
        if (k >= 0 && !v.mask[k]) continue;
        Gen3Game c = m_game;
        try {
            if (k < 0) c.forfeit(); else if (k < 4) c.chooseMove(k); else c.chooseSwitch(k - 4);
        } catch (const std::exception&) { continue; }
        if (c.run(400000) == Gen3Game::BATTLE_OVER && matches(c, snap)) {
            m_game = c;
            obs_finish(m_obs, m_game);
            m_atDecision = false;
            m_phase = Phase::SWAP;
            return true;
        }
    }
    m_mirrorSynced = false;
    m_mirrorError = "the simulator does not reproduce the end of the battle";
    return false;
}

void FactorySession::setContext(int streak, int battle, int challenge, int rents, bool noland, int trainerId) {
    m_ctx = {streak, battle, challenge, rents, noland, trainerId};
}

std::string FactorySession::rentalViewJson(const std::string& mons, int hintType, int hintStyle) {
    Json j;
    j.beginObj().key("candidates").beginArr();
    for (int i = 0; i < 6; i++) ownMon(j, decode_pokemon(reinterpret_cast<const uint8_t*>(mons.data()) + i * 102 + 2), 0);
    j.endArr().key("frontier_ids").beginArr();
    for (int i = 0; i < 6; i++) j.num(rd16(reinterpret_cast<const uint8_t*>(mons.data()) + i * 102));
    j.endArr().kv("hint_type", hintType).kv("hint_style", hintStyle).endObj();
    return j.out;
}

// ---- search ----

std::vector<SearchRoot> FactorySession::searchRoots(const std::vector<std::vector<Gen3Game::DetMon>>& attempts, int R,
                                                    const Gen3Game::DetHidden& hidden,
                                                    const std::vector<uint64_t>& seeds) {
    if (m_phase != Phase::BATTLE && m_phase != Phase::FORCED_SWITCH) throw std::runtime_error("not at a battle decision");
    const bool forced = m_phase == Phase::FORCED_SWITCH;
    const int K = R > 0 ? int(attempts.size()) / R : 0;
    if (K <= 0 || seeds.size() < size_t(2 * K)) throw std::invalid_argument("search roots");
    const uint8_t truthUnusable = m_game.unusableMoves(0);
    const bool truthSwitch = m_game.canSwitch(0);
    m_searchGames.clear();
    std::vector<SearchRoot> roots;
    for (int k = 0; k < K; k++) {
        std::unique_ptr<Gen3Game> g;
        for (int r = 0; r < R; r++) {
            g = std::make_unique<Gen3Game>(m_game);
            g->determinize(attempts[size_t(k) * R + r], &hidden, int64_t(seeds[2 * k] & 0x7FFFFFFFFFFFFFFFULL));
            if (forced || (g->unusableMoves(0) == truthUnusable && g->canSwitch(0) == truthSwitch)) break;
        }
        auto obs = std::make_unique<CppNodeObs>(m_obs);
        obs_rebase(obs->memory(), *g, forced);
        g->redrawTurn(uint32_t(seeds[2 * k + 1]));
        roots.push_back({g.get(), std::move(obs), 0});
        m_searchGames.push_back(std::move(g));
    }
    return roots;
}

void FactorySession::searchBegin(const std::vector<std::vector<Gen3Game::DetMon>>& attempts, int R,
                                 const Gen3Game::DetHidden& hidden, const std::vector<uint64_t>& seeds,
                                 const float rootPriors[7], float rootValue, const SearchConfig& cfg) {
    m_search.reset();
    std::vector<SearchRoot> roots = searchRoots(attempts, R, hidden, seeds);
    bool legal[7];
    for (int a = 0; a < 7; a++) legal[a] = m_obs.view.mask[a];
    Info r = runInfo();
    EncodeCtx ctx{r.streak, r.battle, r.challenge, r.rents};
    set_encode_version(4);
    m_search = std::make_unique<SearchSession>(std::move(roots), ctx, rootPriors, legal, rootValue, cfg);
}

SearchStats FactorySession::searchRunNative(const std::vector<std::vector<Gen3Game::DetMon>>& attempts, int R,
                                            const Gen3Game::DetHidden& hidden, const std::vector<uint64_t>& seeds,
                                            const float rootPriors[7], float rootValue, const SearchConfig& cfg,
                                            const BatchEvaluator& evaluator) {
    m_search.reset();
    std::vector<SearchRoot> roots = searchRoots(attempts, R, hidden, seeds);
    bool legal[7];
    for (int a = 0; a < 7; a++) legal[a] = m_obs.view.mask[a];
    Info r = runInfo();
    EncodeCtx ctx{r.streak, r.battle, r.challenge, r.rents};
    set_encode_version(4);
    SearchStats st = runSearch(roots, ctx, rootPriors, legal, rootValue, cfg, evaluator);
    m_searchGames.clear();
    return st;
}

int FactorySession::searchPrepare() {
    if (!m_search) return -1;
    set_encode_version(4);
    return m_search->prepare();
}
const EncodedObs* FactorySession::searchObservations() const {
    if (!m_search) throw std::runtime_error("no search");
    return m_search->observations();
}
void FactorySession::searchSubmit(const float* priors, const float* values) {
    if (!m_search) throw std::runtime_error("no search");
    m_search->submit(priors, values);
}
SearchStats FactorySession::searchStats() const {
    if (!m_search) throw std::runtime_error("no search");
    return m_search->finished() ? m_search->stats() : m_search->current();
}

// ---- save / load ----
namespace {
constexpr uint32_t SAVE_MAGIC = 0x46534732;   // "2GSF"
struct SessionFields {
    uint32_t magic, stateSize, obsSize;
    int32_t phase, openLevel, maxTurns, turns, startStreak, lastWon, hasDone;
    int32_t mirror, mirrorSynced, atDecision;     // mirror mode (a savestate of the real game resumes its battle)
    int32_t ctx[6];
};
}  // namespace

std::string FactorySession::saveState() {
    std::string game = m_game.stateBytes();
    SessionFields f{SAVE_MAGIC, uint32_t(game.size()), uint32_t(sizeof(ObsMemory)), int32_t(m_phase), m_openLevel,
                    m_maxTurns, m_turns, m_startStreak, m_lastWon, m_hasDone,
                    m_mirror, m_mirrorSynced, m_atDecision,
                    {m_ctx.streak, m_ctx.battle, m_ctx.challenge, m_ctx.rents, m_ctx.noland, m_ctx.trainerId}};
    std::string out(reinterpret_cast<const char*>(&f), sizeof f);
    out.append(reinterpret_cast<const char*>(&m_obs), sizeof m_obs);
    out.append(reinterpret_cast<const char*>(&m_obsDone), sizeof m_obsDone);
    out += game;
    return out;
}

void FactorySession::loadState(const std::string& bytes) {
    SessionFields f;
    if (bytes.size() < sizeof f) throw std::invalid_argument("not a session state");
    std::memcpy(&f, bytes.data(), sizeof f);
    if (f.magic != SAVE_MAGIC || f.obsSize != sizeof(ObsMemory) ||
        bytes.size() != sizeof f + 2 * sizeof(ObsMemory) + f.stateSize)
        throw std::invalid_argument("session state from another build");
    const char* p = bytes.data() + sizeof f;
    m_game.loadStateBytes(std::string(p + 2 * sizeof(ObsMemory), f.stateSize));
    m_game.takeMessages();
    std::memcpy(&m_obs, p, sizeof m_obs);
    std::memcpy(&m_obsDone, p + sizeof m_obs, sizeof m_obsDone);
    m_phase = Phase(f.phase);
    m_openLevel = f.openLevel != 0;
    m_maxTurns = f.maxTurns;
    m_turns = f.turns;
    m_startStreak = f.startStreak;
    m_lastWon = f.lastWon;
    m_hasDone = f.hasDone != 0;
    m_mirror = f.mirror != 0;
    m_mirrorSynced = f.mirrorSynced != 0;
    m_atDecision = f.atDecision != 0;
    m_ctx = {f.ctx[0], f.ctx[1], f.ctx[2], f.ctx[3], f.ctx[4] != 0, f.ctx[5]};
    m_mirrorError.clear();
    m_search.reset();
    m_searchGames.clear();
}

}  // namespace pkmn::web
