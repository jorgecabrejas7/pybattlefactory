// Python bindings of the C++ observer / encoder (include/gen3_observer.hpp): class ObsMemory.
//
//   mem = ObsMemory(hint_type, hint_style)          # or ObsMemory.from_python(backend._observer)
//   mem.observe(game, forced, unusable_mask, can_switch)
//   mem.rebase(game, forced)
//   mem.copy()
//   mem.encode(game, ctx) -> dict                   # same keys / dtypes / shapes as rl.encode.battle (v3)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gen3_observer.hpp"

extern "C" {
#include "../gen3/observer_host.h"
}

#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pkmn;

namespace {

int geti(const py::handle& o, const char* name) { return o.attr(name).cast<int>(); }
bool getb(const py::handle& o, const char* name) { return py::bool_(o.attr(name)); }

template <typename T>
void getv(const py::handle& o, const char* name, T* out, size_t n) {
    py::sequence s = o.attr(name);
    if (static_cast<size_t>(py::len(s)) < n) throw std::runtime_error(std::string("from_python: short list ") + name);
    for (size_t i = 0; i < n; i++) out[i] = s[i].cast<T>();
}

void bytes_into(const py::handle& b, uint8_t* out, size_t n) {
    std::string s = py::bytes(py::reinterpret_borrow<py::object>(b));
    if (s.size() < n) throw std::runtime_error("from_python: short bytes");
    std::memcpy(out, s.data(), n);
}

void conv_bmon(const py::handle& m, ObsBMon& o) {
    o.species = geti(m, "species");
    getv(m, "stats", o.stats, 5);
    getv(m, "moves", o.moves, 4);
    getv(m, "ivs", o.ivs, 6);
    o.ability_num = geti(m, "ability_num");
    getv(m, "stat_stages", o.stat_stages, 8);
    o.ability = geti(m, "ability");
    getv(m, "types", o.types, 2);
    getv(m, "pp", o.pp, 4);
    o.hp = geti(m, "hp");
    o.level = geti(m, "level");
    o.max_hp = geti(m, "max_hp");
    o.item = geti(m, "item");
    o.status1 = m.attr("status1").cast<uint32_t>();
    o.status2 = m.attr("status2").cast<uint32_t>();
}

void conv_pmon(const py::handle& m, ObsPMon& o) {
    o.species = geti(m, "species");
    o.item = geti(m, "held_item");
    getv(m, "moves", o.moves, 4);
    getv(m, "pp", o.pp, 4);
    o.ability_num = geti(m, "ability_num");
    o.level = geti(m, "level");
    o.hp = geti(m, "hp");
    o.max_hp = geti(m, "max_hp");
    getv(m, "stats", o.stats, 5);
    o.status = m.attr("status").cast<uint32_t>();
}

void conv_snap(const py::handle& s, ObsSnap& o) {
    std::memset(&o, 0, sizeof(o));
    py::sequence mons = s.attr("mons");
    for (int b = 0; b < 2; b++) conv_bmon(mons[b], o.mons[b]);
    getv(s, "idx", o.idx, 2);
    py::sequence parties = s.attr("parties");
    for (int side = 0; side < 2; side++) {
        py::sequence p = parties[side];
        for (int i = 0; i < 3; i++) conv_pmon(p[i], o.parties[side][i]);
    }
    py::sequence dis = s.attr("dis");
    for (int b = 0; b < 2; b++) bytes_into(dis[b], o.dis[b], 0x1C);
    getv(s, "status3", o.status3, 2);
    getv(s, "sides", o.sides, 2);
    bytes_into(s.attr("timers"), o.timers, 24);
    o.weather = static_cast<uint16_t>(geti(s, "weather"));
    bytes_into(s.attr("wfk"), o.wfk, 44);
    getv(s, "last_moves", o.last_moves, 2);
    getv(s, "last_printed", o.last_printed, 2);
    getv(s, "bide", o.bide, 2);
    o.crit = geti(s, "crit");
    getv(s, "locked", o.locked, 2);
    o.turn = geti(s, "turn");
    getv(s, "last_used", o.last_used, 2);
    o.player_switches = geti(s, "player_switches");
    o.rnd_turn = geti(s, "rnd_turn");
    uint8_t order[2];
    bytes_into(s.attr("order"), order, 2);
    o.order[0] = order[0];
    o.order[1] = order[1];
    o.forced = getb(s, "forced");
}

void conv_events(const py::handle& e, ObsEvents& o) {
    o.turn = geti(e, "turn");
    o.turns = geti(e, "turns");
    o.in_progress = getb(e, "in_progress");
    o.first = geti(e, "first");
    o.own_action = geti(e, "own_action");
    o.own_move = geti(e, "own_move");
    o.own_crit = geti(e, "own_crit");
    o.enemy_action = geti(e, "enemy_action");
    o.enemy_move = geti(e, "enemy_move");
    o.enemy_crit = geti(e, "enemy_crit");
    o.damage_dealt_pixels = geti(e, "damage_dealt_pixels");
    o.damage_taken = geti(e, "damage_taken");
    o.own_fainted = getb(e, "own_fainted");
    o.enemy_fainted = getb(e, "enemy_fainted");
}

const char* const ACTIVE_BOOLS[26] = {"confused", "infatuated", "substitute", "leech_seeded", "cursed", "nightmare",
                                      "trapped", "focus_energy", "transformed", "perish_song", "rooted", "yawn",
                                      "torment", "taunted", "must_recharge", "destiny_bond", "defense_curl",
                                      "foresight", "minimized", "charged_up", "imprisoning", "grudge", "mud_sport",
                                      "water_sport", "rage", "first_turn"};
const char* const ACTIVE_COUNTERS[16] = {"confusion_turns", "toxic_counter", "taunt_turns_left", "encore_turns",
                                         "disable_turns", "yawn_turns_left", "wrapped_turns", "uproar_turns",
                                         "rampage_turns", "bide_turns_left", "rollout_hits_left", "fury_cutter_count",
                                         "stockpile", "charge_turns_left", "locked_on_turns_left", "protect_uses"};

void conv_active(const py::handle& a, ObsActive& o) {
    o.party_index = geti(a, "party_index");
    getv(a, "stat_stages", o.stat_stages, 7);
    getv(a, "types", o.types, 2);
    for (int k = 0; k < 26; k++) o.bools[k] = getb(a, ACTIVE_BOOLS[k]);
    for (int k = 0; k < 16; k++) o.counters[k] = geti(a, ACTIVE_COUNTERS[k]);
    o.perish_count = geti(a, "perish_count");
    o.substitute_hp = geti(a, "substitute_hp");
    o.semi_invulnerable = geti(a, "semi_invulnerable");
    o.encored_move = geti(a, "encored_move");
    o.disabled_move = geti(a, "disabled_move");
    o.locked_move = geti(a, "locked_move");
    o.charging_move = geti(a, "charging_move");
}

void conv_side(const py::handle& s, ObsSide& o) {
    o.reflect_turns = geti(s, "reflect_turns");
    o.light_screen_turns = geti(s, "light_screen_turns");
    o.safeguard_turns = geti(s, "safeguard_turns");
    o.mist_turns = geti(s, "mist_turns");
    o.spikes = geti(s, "spikes");
    o.future_sight_turns = geti(s, "future_sight_turns");
    o.future_sight_move = geti(s, "future_sight_move");
    o.wish_turns = geti(s, "wish_turns");
}

int list_into(const py::handle& lst, int32_t* out, int cap, bool& overflow) {
    py::sequence s = py::reinterpret_borrow<py::sequence>(lst);
    int n = static_cast<int>(py::len(s));
    if (n > cap) {
        overflow = true;
        n = cap;
    }
    for (int i = 0; i < n; i++) out[i] = s[i].cast<int>();
    return n;
}

void conv_view(const py::handle& v, ObsView& o, bool& overflow) {
    std::memset(&o, 0, sizeof(o));
    o.valid = true;
    py::sequence own = v.attr("own_party");
    for (int i = 0; i < 3; i++) {
        py::handle m = own[i];
        ObsOwnMon& w = o.own[i];
        w.species = geti(m, "species");
        w.level = geti(m, "level");
        w.hp = geti(m, "hp");
        w.max_hp = geti(m, "max_hp");
        getv(m, "stats", w.stats, 5);
        getv(m, "moves", w.moves, 4);
        getv(m, "pp", w.pp, 4);
        w.item = geti(m, "item");
        w.ability = geti(m, "ability");
        w.status = geti(m, "status");
        getv(m, "types", w.types, 2);
        w.sleep_turns = geti(m, "sleep_turns");
    }
    py::sequence enemy = v.attr("enemy_party");
    for (int i = 0; i < 3; i++) {
        py::handle m = enemy[i];
        ObsSeenMon& w = o.enemy[i];
        w.seen = getb(m, "seen");
        w.fainted = getb(m, "fainted");
        w.species = geti(m, "species");
        w.level = geti(m, "level");
        w.hp_pixels = geti(m, "hp_pixels");
        w.status = geti(m, "status");
        w.n_moves = list_into(m.attr("revealed_moves"), w.moves, OBS_MAX_REVEALED, overflow);
        py::object item = m.attr("revealed_item");
        w.item = item.is_none() ? -1 : item.cast<int>();
        w.n_types = list_into(m.attr("types"), w.types, 2, overflow);
        w.n_base = list_into(m.attr("base_stats"), w.base_stats, 6, overflow);
        w.n_abilities = list_into(m.attr("possible_abilities"), w.possible_abilities, 2, overflow);
        w.revealed_ability = geti(m, "revealed_ability");
        w.sleep_turns = geti(m, "sleep_turns");
    }
    conv_active(v.attr("own_active"), o.own_active);
    conv_active(v.attr("enemy_active"), o.enemy_active);
    o.weather = geti(v, "weather");
    o.weather_permanent = getb(v, "weather_permanent");
    o.weather_turns_left = geti(v, "weather_turns_left");
    conv_side(v.attr("own_side"), o.own_side);
    conv_side(v.attr("enemy_side"), o.enemy_side);
    o.turn = geti(v, "turn");
    o.forced_switch = getb(v, "forced_switch");
    o.must_struggle = getb(v, "must_struggle");
    getv(v, "usable_moves", o.usable, 4);
    for (py::handle a : v.attr("legal_actions")) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(a);
        std::string kind = t[0].cast<std::string>();
        if (kind == "move") o.mask[t[1].cast<int>()] = true;
        else if (kind == "switch") o.mask[4 + t[1].cast<int>()] = true;
    }
    conv_events(v.attr("last_turn"), o.last_turn);
    o.hint_type = geti(v, "hint_type");
    o.hint_style = geti(v, "hint_style");
}

// A Python BattleObserver's whole memory -> ObsMemory
ObsMemory from_python(const py::object& ob) {
    ObsMemory m;
    obs_init(m, static_cast<uint16_t>(geti(ob, "hint_type")), static_cast<uint16_t>(geti(ob, "hint_style")));
    auto idx3 = [&](const py::handle& k) {
        int i = k.cast<int>();
        if (i < 0 || i >= 3) {
            m.overflow = true;
            return -1;
        }
        return i;
    };
    for (auto kv : ob.attr("revealed_moves").cast<py::dict>()) {
        int i = idx3(kv.first);
        if (i < 0) continue;
        bool of = false;
        m.n_revealed[i] = static_cast<uint8_t>(list_into(kv.second, m.revealed_moves[i], OBS_MAX_REVEALED, of));
        m.overflow = m.overflow || of;
    }
    for (auto kv : ob.attr("revealed_items").cast<py::dict>())
        if (int i = idx3(kv.first); i >= 0) m.revealed_items[i] = kv.second.cast<int>();
    for (auto kv : ob.attr("revealed_abilities").cast<py::dict>())
        if (int i = idx3(kv.first); i >= 0) m.revealed_abilities[i] = kv.second.cast<int>();
    for (py::handle x : ob.attr("seen")) {
        int i = x.cast<int>();
        if (i >= 0 && i < 8) m.seen |= static_cast<uint8_t>(1u << i);
        else m.overflow = true;
    }
    for (auto kv : ob.attr("_last_enemy_item").cast<py::dict>())
        if (int i = idx3(kv.first); i >= 0) m.last_enemy_item[i] = kv.second.cast<int>();
    for (auto kv : ob.attr("_bench_status").cast<py::dict>())
        if (int i = idx3(kv.first); i >= 0) m.bench_status[i] = kv.second.cast<int>();
    for (auto kv : ob.attr("_sleep").cast<py::dict>()) {
        py::tuple k = kv.first.cast<py::tuple>();
        int side = k[0].cast<int>(), i = k[1].cast<int>();
        if (side < 0 || side > 1 || i < 0 || i >= 3) {
            m.overflow = true;
            continue;
        }
        py::sequence rec = kv.second.cast<py::sequence>();
        m.sleep[side][i] = {true, rec[0].cast<int>(), rec[1].cast<int>()};
    }
    static const char* const VOL[5] = {"encore", "disable", "wrapped", "uproar", "rampage"};
    py::sequence vols = ob.attr("_vol");
    for (int b = 0; b < 2; b++) {
        py::dict d = vols[b].cast<py::dict>();
        ObsVol& v = m.vol[b];
        if (d.contains("mon")) {
            v.has_mon = true;
            v.mon = d["mon"].cast<int>();
        }
        if (d.contains("confusion")) {
            py::sequence c = d["confusion"].cast<py::sequence>();
            v.has_conf = true;
            v.conf_tried = c[0].cast<int>();
            v.conf_raw = c[1].cast<int>();
        }
        for (int k = 0; k < 5; k++)
            if (d.contains(VOL[k])) {
                v.has[k] = true;
                v.start[k] = d[VOL[k]].cast<int>();
            }
    }
    m.turns_done = geti(ob, "_turns_done");
    if (py::hasattr(ob, "_records")) {
        py::sequence recs = ob.attr("_records");
        for (int j = 0; j < 3 && j < (int)py::len(recs); j++) {
            py::sequence r = recs[j];
            for (int k = 0; k < 6; k++) m.records[j][k] = r[k].cast<int>();
        }
        m.team_max_hp = geti(ob, "_team_max_hp");
        m.finished = getb(ob, "_finished");
    }
    py::object start = ob.attr("_start"), prev = ob.attr("_prev");
    if (!start.is_none()) {
        conv_snap(start, m.start);
        m.has_start = true;
    }
    if (!prev.is_none()) {
        conv_snap(prev, m.prev);
        m.has_prev = true;
    }
    conv_events(ob.attr("_events"), m.events);
    py::object key = ob.attr("_cache_key");
    if (!key.is_none()) {
        py::tuple k = key.cast<py::tuple>();
        py::tuple sk = k[0].cast<py::tuple>();          // (forced, raw, idx, turn, last, rnd_turn)
        m.cache.valid = true;
        m.cache.forced = py::bool_(sk[0]);
        bytes_into(sk[1], m.cache.raw, sizeof(m.cache.raw));
        bytes_into(sk[2], m.cache.idx, 4);
        m.cache.turn = sk[3].cast<int>();
        bytes_into(sk[4], m.cache.last, 4);
        m.cache.rnd_turn = sk[5].cast<int>();
        m.cache.unusable = k[1].cast<int>();
        m.cache.can_switch = py::bool_(k[2]);
    }
    py::object view = ob.attr("_cache_view");
    if (!view.is_none()) conv_view(view, m.view, m.overflow);
    return m;
}

template <typename T, size_t N>
py::array_t<T> arr(const T* data, const std::array<py::ssize_t, N>& shape) {
    py::array_t<T> a(std::vector<py::ssize_t>(shape.begin(), shape.end()));
    std::memcpy(a.mutable_data(), data, a.size() * sizeof(T));
    return a;
}

EncodeCtx to_ctx(const py::dict& d) {
    return {d["streak"].cast<int>(), d["battle"].cast<int>(), d["challenge"].cast<int>(), d["rents"].cast<int>()};
}

py::dict to_dict(const EncodedObs& e) {
    py::dict d;
    d["mon_ids"] = arr<int64_t, 2>(&e.mon_ids[0][0], {6, MON_IDS});
    d["mon_num"] = arr<float, 2>(e.mon_num, {6, e.mon_w});
    d["move_num"] = arr<float, 3>(e.move_num, {6, 4, e.move_w});
    d["ctx_ids"] = arr<int64_t, 1>(e.ctx_ids, {CTX_IDS});
    d["ctx_num"] = arr<float, 1>(e.ctx_num, {e.ctx_w});
    d["mask"] = arr<bool, 1>(e.mask, {7});
    d["active"] = py::module_::import("numpy").attr("int64")(e.active);
    return d;
}

}  // namespace

void bind_observer(py::module_& m) {
    py::class_<ObsMemory>(m, "ObsMemory",
                          "The C++ BattleObserver (pybattle/view.py) + rl.encode.battle v3 / v4 (include/gen3_observer.hpp)")
        .def(py::init([](int hint_type, int hint_style) {
                 ObsMemory mem;
                 obs_init(mem, static_cast<uint16_t>(hint_type), static_cast<uint16_t>(hint_style));
                 return mem;
             }),
             py::arg("hint_type") = 18, py::arg("hint_style") = 0)
        .def("observe", [](ObsMemory& mem, Gen3Game& g, bool forced, int unusable, bool canSwitch) {
                 obs_observe(mem, g, forced, static_cast<uint8_t>(unusable), canSwitch);
             }, py::arg("game"), py::arg("forced"), py::arg("unusable_mask"), py::arg("can_switch") = true)
        .def("rebase", [](ObsMemory& mem, Gen3Game& g, bool forced) { obs_rebase(mem, g, forced); },
             py::arg("game"), py::arg("forced"))
        .def("finish", [](ObsMemory& mem, Gen3Game& g) { obs_finish(mem, g); }, py::arg("game"),
             "BattleObserver.finish: the battle has just been decided (before the game winds it down)")
        .def_property_readonly("records", [](const ObsMemory& mem) {
            // BattleObserver._records: per enemy party slot [damage, knockouts, turns, hits_taken, max_boosts,
            // inflicted_status]
            std::vector<std::vector<int>> out(3);
            for (int j = 0; j < 3; j++) out[j].assign(mem.records[j], mem.records[j] + 6);
            return out;
        })
        .def_property_readonly("team_max_hp", [](const ObsMemory& mem) { return mem.team_max_hp; })
        .def_property_readonly("finished", [](const ObsMemory& mem) { return mem.finished; })
        .def_property_readonly("revealed_moves", [](const ObsMemory& mem) {
            // BattleObserver.revealed_moves (slots with at least one revealed move)
            py::dict d;
            for (int i = 0; i < 3; i++)
                if (mem.n_revealed[i])
                    d[py::int_(i)] = std::vector<int>(mem.revealed_moves[i], mem.revealed_moves[i] + mem.n_revealed[i]);
            return d;
        })
        .def_property_readonly("revealed_items", [](const ObsMemory& mem) {
            py::dict d;
            for (int i = 0; i < 3; i++)
                if (mem.revealed_items[i] >= 0) d[py::int_(i)] = mem.revealed_items[i];
            return d;
        })
        .def_property_readonly("revealed_abilities", [](const ObsMemory& mem) {
            py::dict d;
            for (int i = 0; i < 3; i++)
                if (mem.revealed_abilities[i]) d[py::int_(i)] = mem.revealed_abilities[i];
            return d;
        })
        .def("copy", [](const ObsMemory& mem) { return ObsMemory(mem); })
        .def("__copy__", [](const ObsMemory& mem) { return ObsMemory(mem); })
        .def("encode", [](const ObsMemory& mem, Gen3Game& g, const py::dict& ctx) {
                 if (!mem.view.valid) throw std::runtime_error("ObsMemory.encode before any observe");
                 EncodedObs e;
                 encode_battle(mem, g, to_ctx(ctx), e);
                 return to_dict(e);
             }, py::arg("game"), py::arg("ctx"))
        .def("bench", [](const ObsMemory& mem, Gen3Game& g, bool forced, int unusable, bool canSwitch,
                         const py::dict& ctx, int reps) {
                 // µs per (copy + observe) and per encode, timed in C++ (no Python in the loop)
                 EncodeCtx c = to_ctx(ctx);
                 EncodedObs e;
                 double tObs = 0, tEnc = 0;
                 for (int r = 0; r < reps; r++) {
                     auto t0 = std::chrono::steady_clock::now();
                     ObsMemory m2 = mem;
                     obs_observe(m2, g, forced, static_cast<uint8_t>(unusable), canSwitch);
                     auto t1 = std::chrono::steady_clock::now();
                     encode_battle(m2, g, c, e);
                     auto t2 = std::chrono::steady_clock::now();
                     tObs += std::chrono::duration<double, std::micro>(t1 - t0).count();
                     tEnc += std::chrono::duration<double, std::micro>(t2 - t1).count();
                 }
                 return py::make_tuple(tObs / reps, tEnc / reps);
             }, py::arg("game"), py::arg("forced"), py::arg("unusable_mask"), py::arg("can_switch"), py::arg("ctx"),
             py::arg("reps") = 100)
        .def_static("from_python", &from_python, py::arg("observer"),
                    "Convert a pybattle.view.BattleObserver's whole memory (any backend).")
        .def_property_readonly("overflow", [](const ObsMemory& mem) { return mem.overflow; })
        .def_property_readonly("turns_done", [](const ObsMemory& mem) { return mem.turns_done; })
        .def_property_readonly("seen", [](const ObsMemory& mem) {
            std::vector<int> out;
            for (int i = 0; i < 8; i++)
                if (mem.seen & (1u << i)) out.push_back(i);
            return out;
        })
        .def_property_readonly_static("nbytes", [](py::object) { return sizeof(ObsMemory); });

    m.def("set_encode_version", &set_encode_version, py::arg("version"),
          "Encoding version of ObsMemory.encode and the C++ search (3 or 4); rl.encode.set_version calls it.");
    m.def("encode_version", &encode_version);

    // The decomp tables the C++ observer uses (tests compare them with game_data.json)
    m.def("obs_tables", []() {
        const Gen3ObsTables* t = Gen3Obs_Tables();
        py::list species, moves, items;
        for (int i = 0; i < GEN3_OBS_NUM_SPECIES; i++) {
            const auto& s = t->species[i];
            species.append(py::make_tuple(std::vector<int>(s.base, s.base + 6), std::vector<int>(s.types, s.types + 2),
                                          std::vector<int>(s.abilities, s.abilities + 2)));
        }
        for (int i = 0; i < GEN3_OBS_MOVES_COUNT; i++) {
            const auto& v = t->moves[i];
            py::dict d;
            d["effect"] = v.effect; d["power"] = v.power; d["type"] = v.type; d["accuracy"] = v.accuracy;
            d["pp"] = v.pp; d["secondary_chance"] = v.secondaryChance; d["flags"] = v.flags;
            d["priority"] = v.priority;
            moves.append(d);
        }
        for (int i = 0; i < GEN3_OBS_ITEMS_COUNT; i++)
            items.append(py::make_tuple(t->items[i].holdEffect, t->items[i].holdEffectParam));
        py::dict out;
        out["species"] = species;
        out["moves"] = moves;
        out["items"] = items;
        out["type_chart"] = std::vector<int>(t->typeChart, t->typeChart + GEN3_OBS_TYPE_CHART);
        py::list ratios;
        for (auto& r : t->statStageRatios) ratios.append(py::make_tuple(r[0], r[1]));
        out["stat_stage_ratios"] = ratios;
        out["abilities_count"] = t->abilitiesCount;
        out["effects_count"] = t->effectsCount;
        return out;
    });
}
