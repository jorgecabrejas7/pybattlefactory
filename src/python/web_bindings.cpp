// FactorySession (src/web/factory_session.hpp) for tests: tests/python/test_web_session.py checks it
// against pybattle.backend.SimBackend.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../web/factory_session.hpp"

#include <cstring>

namespace py = pybind11;
using namespace pkmn;
using namespace pkmn::web;

namespace {
// The packed search inputs of fs_search_begin (src/web/web_api.cpp).
std::vector<std::vector<Gen3Game::DetMon>> unpackAttempts(const py::array_t<int32_t, py::array::c_style>& att) {
    const int32_t* p = att.data();
    const size_t n = size_t(att.size()) / 66;
    std::vector<std::vector<Gen3Game::DetMon>> out(n);
    for (auto& a : out) {
        for (int s = 0; s < 3; s++) {
            Gen3Game::DetMon m{};
            m.partySlot = p[0];
            m.species = p[1];
            for (int i = 0; i < 4; i++) m.moves[i] = p[2 + i];
            m.item = p[6];
            for (int i = 0; i < 6; i++) m.ivs[i] = p[7 + i];
            for (int i = 0; i < 6; i++) m.evs[i] = p[13 + i];
            m.nature = p[19];
            m.abilityBit = p[20];
            std::memcpy(&m.hpFraction, &p[21], 4);
            a.push_back(m);
            p += 22;
        }
    }
    return out;
}
Gen3Game::DetHidden unpackHidden(const py::array_t<int32_t, py::array::c_style>& h) {
    const int32_t* hid = h.data();
    Gen3Game::DetHidden d{};
    for (int b = 0; b < 2; b++)
        for (int i = 0; i < 3; i++) d.sleepElapsed[b][i] = hid[b * 3 + i];
    for (int b = 0; b < 2; b++) {
        d.confusionElapsed[b] = hid[6 + b];
        d.wrapElapsed[b] = hid[8 + b];
        d.uproarElapsed[b] = hid[10 + b];
        d.rampageElapsed[b] = hid[12 + b];
    }
    return d;
}
py::dict encodedBatch(const EncodedObs* e, int n) {
    py::dict d;
    if (n <= 0) return d;
    const int mw = e[0].mon_w, vw = e[0].move_w, cw = e[0].ctx_w;
    py::array_t<int64_t> ids({n, 6, MON_IDS}), cids({n, CTX_IDS}), act({n});
    py::array_t<float> mon({n, 6, mw}), mv({n, 6, 4, vw}), ctx({n, cw});
    py::array_t<bool> mask({n, 7});
    for (int i = 0; i < n; i++) {
        std::memcpy(ids.mutable_data(i), e[i].mon_ids, sizeof e[i].mon_ids);
        std::memcpy(cids.mutable_data(i), e[i].ctx_ids, sizeof e[i].ctx_ids);
        act.mutable_at(i) = e[i].active;
        std::memcpy(mon.mutable_data(i), e[i].mon_num, sizeof(float) * 6 * mw);
        std::memcpy(mv.mutable_data(i), e[i].move_num, sizeof(float) * 24 * vw);
        std::memcpy(ctx.mutable_data(i), e[i].ctx_num, sizeof(float) * cw);
        std::memcpy(mask.mutable_data(i), e[i].mask, sizeof e[i].mask);
    }
    d["mon_ids"] = ids; d["mon_num"] = mon; d["move_num"] = mv; d["ctx_ids"] = cids; d["ctx_num"] = ctx;
    d["mask"] = mask; d["active"] = act;
    return d;
}
py::dict statsDict(const SearchStats& st) {
    py::dict d;
    d["visits"] = std::vector<double>(st.visits, st.visits + 7);
    d["q"] = std::vector<double>(st.q, st.q + 7);
    d["leaves"] = st.leaves;
    d["nodes"] = st.nodes;
    d["errors"] = st.errors;
    d["max_depth"] = st.maxDepth;
    return d;
}
SearchConfig searchCfg(int nSims, int batch, float cPuct) {
    SearchConfig c;
    c.nSims = nSims;
    c.batch = batch;
    c.cPuct = cPuct;
    return c;
}
}  // namespace

void bind_web(py::module_& m) {
    py::class_<FactorySession>(m, "FactorySession")
        .def(py::init<bool, int>(), py::arg("open_level") = true, py::arg("max_turns") = 500)
        .def("reset", &FactorySession::reset, py::arg("seed"), py::arg("win_streak") = 0, py::arg("rents_count") = 0)
        .def_property_readonly("phase", [](const FactorySession& s) { return int(s.phase()); })
        .def("view_json", &FactorySession::viewJson)
        .def("run_info_json", &FactorySession::runInfoJson)
        .def("take_log_json", &FactorySession::takeLogJson)
        .def("estimates_json", &FactorySession::estimatesJson)
        .def("act_rental", &FactorySession::actRental)
        .def("act_swap", &FactorySession::actSwap)
        .def("act_move", &FactorySession::actMove)
        .def("act_switch", &FactorySession::actSwitch)
        .def("act_forfeit", &FactorySession::actForfeit)
        .def_property_readonly("wins", &FactorySession::wins)
        .def_property_readonly("last_battle_won", &FactorySession::lastBattleWon)
        .def("mirror_regions", [](FactorySession& s, const std::vector<std::pair<uint32_t, uint32_t>>& r) {
            std::vector<FactorySession::Region> v;
            for (auto& [a, n] : r) v.push_back({a, n});
            s.mirrorRegions(v);
        })
        .def("mirror_begin", [](FactorySession& s, const py::dict& d, int ht, int hs) {
            FactorySession::BattleStart st{};
            st.rng = d["rng"].cast<uint32_t>();
            st.battleTypeFlags = d["battle_type_flags"].cast<uint32_t>();
            st.trainerId = d["trainer_id"].cast<uint16_t>();
            st.lvlMode = d["lvl_mode"].cast<uint8_t>();
            st.battleNum = d["battle_num"].cast<uint16_t>();
            auto f = d["factory_win_streaks"].cast<std::vector<uint16_t>>();
            auto t = d["tower_win_streaks"].cast<std::vector<uint16_t>>();
            for (int i = 0; i < 4; i++) st.factoryStreaks[i] = f[i];
            for (int i = 0; i < 8; i++) st.towerStreaks[i] = t[i];
            st.playerParty = std::string(d["player_party"].cast<py::bytes>());
            st.enemyParty = std::string(d["enemy_party"].cast<py::bytes>());
            if (d.contains("rental_mons")) st.rentalMons = std::string(d["rental_mons"].cast<py::bytes>());
            s.mirrorBegin(st, ht, hs);
        })
        .def("mirror_sync", [](FactorySession& s, bool forced, const py::bytes& b) { return s.mirrorSync(forced, std::string(b)) != 0; })
        .def("mirror_finish", [](FactorySession& s, const py::bytes& b) { return s.mirrorFinish(std::string(b)); })
        .def_property_readonly("mirror_error", &FactorySession::mirrorError)
        .def("set_context", &FactorySession::setContext)
        .def("rental_view_json", [](FactorySession& s, const py::bytes& b, int ht, int hs) { return s.rentalViewJson(std::string(b), ht, hs); })
        .def("swap_view_json", [](FactorySession& s, const py::bytes& o, const py::bytes& e, int ht, int hs) {
            return s.swapViewJson(std::string(o), std::string(e), ht, hs);
        })
        .def("search_begin", [](FactorySession& s, const py::array_t<int32_t, py::array::c_style>& att, int R,
                                const py::array_t<int32_t, py::array::c_style>& hid, const std::vector<uint64_t>& seeds,
                                const std::vector<float>& priors, float value, int nSims, int batch, float cPuct) {
            s.searchBegin(unpackAttempts(att), R, unpackHidden(hid), seeds, priors.data(), value, searchCfg(nSims, batch, cPuct));
        })
        .def("search_prepare", &FactorySession::searchPrepare)
        .def("search_observations", [](FactorySession& s, int n) { return encodedBatch(s.searchObservations(), n); })
        .def("search_submit", [](FactorySession& s, const py::array_t<float, py::array::c_style>& p,
                                 const py::array_t<float, py::array::c_style>& v) { s.searchSubmit(p.data(), v.data()); })
        .def("search_stats", [](FactorySession& s) { return statsDict(s.searchStats()); })
        .def("search_end", &FactorySession::searchEnd)
        .def("search_run_native", [](FactorySession& s, const py::array_t<int32_t, py::array::c_style>& att, int R,
                                     const py::array_t<int32_t, py::array::c_style>& hid, const std::vector<uint64_t>& seeds,
                                     const std::vector<float>& priors, float value, int nSims, int batch, float cPuct,
                                     const py::function& evaluate) {
            BatchEvaluator ev = [&](const EncodedObs* obs, int n, float* pr, float* va) {
                py::tuple r = evaluate(encodedBatch(obs, n));
                auto P = r[0].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                auto V = r[1].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                std::memcpy(pr, P.data(), sizeof(float) * 7 * n);
                std::memcpy(va, V.data(), sizeof(float) * n);
            };
            return statsDict(s.searchRunNative(unpackAttempts(att), R, unpackHidden(hid), seeds, priors.data(), value,
                                               searchCfg(nSims, batch, cPuct), ev));
        })
        .def("save_state", [](FactorySession& s) { return py::bytes(s.saveState()); })
        .def("load_state", [](FactorySession& s, const py::bytes& b) { s.loadState(std::string(b)); })
        .def("encode_battle", [](FactorySession& s) {
            const EncodedObs& e = s.encodeBattle();
            py::dict d;
            py::array_t<int64_t> ids({6, MON_IDS});
            std::memcpy(ids.mutable_data(), e.mon_ids, sizeof e.mon_ids);
            py::array_t<float> mon({6, e.mon_w}), mv({6, 4, e.move_w}), ctx({e.ctx_w});
            std::memcpy(mon.mutable_data(), e.mon_num, sizeof(float) * 6 * e.mon_w);
            std::memcpy(mv.mutable_data(), e.move_num, sizeof(float) * 24 * e.move_w);
            std::memcpy(ctx.mutable_data(), e.ctx_num, sizeof(float) * e.ctx_w);
            py::array_t<int64_t> cids({CTX_IDS});
            std::memcpy(cids.mutable_data(), e.ctx_ids, sizeof e.ctx_ids);
            py::array_t<bool> mask({7});
            std::memcpy(mask.mutable_data(), e.mask, sizeof e.mask);
            d["mon_ids"] = ids;
            d["mon_num"] = mon;
            d["move_num"] = mv;
            d["ctx_ids"] = cids;
            d["ctx_num"] = ctx;
            d["mask"] = mask;
            d["active"] = e.active;
            return d;
        });
}
