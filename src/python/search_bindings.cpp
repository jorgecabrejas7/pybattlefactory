// Python bindings of the C++ search loop (include/gen3_search.hpp): Searcher, and a node observer
// that calls the Python BattleObserver + rl.encode.battle (used until / unless the C++ ObsMemory
// is available, and to test the loop against the Python search with the very same observer).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "gen3_search.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace pkmn;

namespace {

template <typename T>
void copyField(const py::dict& d, const char* key, T* dst, size_t n) {
    auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(d[key]);
    if (!a || static_cast<size_t>(a.size()) != n)
        throw std::runtime_error(std::string("encode.battle: bad array '") + key + "'");
    std::memcpy(dst, a.data(), n * sizeof(T));
}

// A numeric array of any encoding version: `rows` rows of a width that fits `max_w`; returns the width.
int32_t copyPacked(const py::dict& d, const char* key, float* dst, size_t rows, int max_w) {
    auto a = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(d[key]);
    if (!a || a.size() % rows != 0 || static_cast<size_t>(a.size()) / rows > static_cast<size_t>(max_w))
        throw std::runtime_error(std::string("encode.battle: bad array '") + key + "'");
    std::memcpy(dst, a.data(), a.size() * sizeof(float));
    return static_cast<int32_t>(a.size() / rows);
}

// BattleObserver (pybattle/view.py) + rl.encode.battle, through Python.
class PyNodeObs : public NodeObs {
public:
    PyNodeObs(py::object observer, py::object encodeFn, py::object ctx)
        : m_obs(std::move(observer)), m_encode(std::move(encodeFn)), m_ctx(std::move(ctx)) {}
    std::unique_ptr<NodeObs> clone() const override {
        return std::make_unique<PyNodeObs>(m_obs.attr("fast_copy")(), m_encode, m_ctx);
    }
    void observe(Gen3Game& g, bool forced, uint8_t unusable, bool canSwitch) override {
        py::object game = py::cast(&g, py::return_value_policy::reference);
        m_view = m_obs.attr("observe")(game, forced, unusable, canSwitch);
    }
    void encode(Gen3Game&, const EncodeCtx&, EncodedObs& out) override {
        if (m_view.is_none()) throw std::runtime_error("encode before observe");
        py::dict d = m_encode(m_view, m_ctx);
        copyField<int64_t>(d, "mon_ids", &out.mon_ids[0][0], 6 * MON_IDS);
        out.mon_w = copyPacked(d, "mon_num", out.mon_num, 6, MON_NUM_MAX);
        out.move_w = copyPacked(d, "move_num", out.move_num, 6 * 4, MOVE_NUM_MAX);
        copyField<int64_t>(d, "ctx_ids", out.ctx_ids, CTX_IDS);
        out.ctx_w = copyPacked(d, "ctx_num", out.ctx_num, 1, CTX_NUM_MAX);
        copyField<bool>(d, "mask", out.mask, 7);
        copyField<int64_t>(d, "active", &out.active, 1);
    }

private:
    py::object m_obs, m_encode, m_ctx, m_view = py::none();
};

template <typename T>
py::array_t<T> stacked(const EncodedObs* obs, int n, std::vector<py::ssize_t> shape, size_t offset, size_t count) {
    std::vector<py::ssize_t> full{n};
    full.insert(full.end(), shape.begin(), shape.end());
    py::array_t<T> a(full);
    T* dst = a.mutable_data();
    for (int i = 0; i < n; i++)
        std::memcpy(dst + static_cast<size_t>(i) * count, reinterpret_cast<const char*>(&obs[i]) + offset,
                    count * sizeof(T));
    return a;
}

// The batch as rl.encode.collate stacks it (before torch): contiguous numpy arrays.
py::dict batchDict(const EncodedObs* obs, int n) {
    py::dict d;
    // one layout per batch (the encoding version does not change during a search)
    const py::ssize_t mw = n ? obs[0].mon_w : MON_NUM_V3, vw = n ? obs[0].move_w : MOVE_NUM,
                      cw = n ? obs[0].ctx_w : CTX_NUM;
    for (int i = 1; i < n; i++)
        if (obs[i].mon_w != mw || obs[i].move_w != vw || obs[i].ctx_w != cw)
            throw std::runtime_error("observations of different encoding layouts in one batch");
    d["mon_ids"] = stacked<int64_t>(obs, n, {6, MON_IDS}, offsetof(EncodedObs, mon_ids), 6 * MON_IDS);
    d["mon_num"] = stacked<float>(obs, n, {6, mw}, offsetof(EncodedObs, mon_num), 6 * mw);
    d["move_num"] = stacked<float>(obs, n, {6, 4, vw}, offsetof(EncodedObs, move_num), 6 * 4 * vw);
    d["ctx_ids"] = stacked<int64_t>(obs, n, {CTX_IDS}, offsetof(EncodedObs, ctx_ids), CTX_IDS);
    d["ctx_num"] = stacked<float>(obs, n, {cw}, offsetof(EncodedObs, ctx_num), cw);
    d["mask"] = stacked<bool>(obs, n, {7}, offsetof(EncodedObs, mask), 7);
    d["active"] = stacked<int64_t>(obs, n, {}, offsetof(EncodedObs, active), 1);
    return d;
}

EncodeCtx ctxFrom(const py::handle& ctx) {
#if PKMN_HAVE_OBSERVER
    if (py::isinstance<EncodeCtx>(ctx)) return ctx.cast<EncodeCtx>();
#endif
    py::dict d = py::reinterpret_borrow<py::dict>(ctx);
    EncodeCtx c{};
    c.streak = d["streak"].cast<int>();
    c.battle = d["battle"].cast<int>();
    c.challenge = d["challenge"].cast<int>();
    c.rents = d["rents"].cast<int>();
    return c;
}

py::dict ctxDict(const EncodeCtx& c) {
    py::dict d;
    d["streak"] = c.streak;
    d["battle"] = c.battle;
    d["challenge"] = c.challenge;
    d["rents"] = c.rents;
    return d;
}

struct Searcher {
    SearchConfig cfg;

    py::dict search(const py::list& roots, const py::object& ctx, const std::vector<float>& rootPriors,
                    const std::vector<bool>& rootLegal, float rootValue, const py::function& evaluator) {
        if (rootPriors.size() != 7 || rootLegal.size() != 7)
            throw std::invalid_argument("root priors and legal need 7 entries");
        // No true hidden information in training (rl/search.py mark_training): every root must be a full
        // determinization (Gen3Game.determinize), whoever built the roots.
        const char* training = std::getenv("PYB_TRAINING");
        if (training && *training) {
            for (const py::handle& item : roots) {
                py::tuple t = py::reinterpret_borrow<py::tuple>(item);
                if (t.size() < 1 || !t[0].cast<Gen3Game&>().determinized()) {
                    PyErr_SetString(PyExc_PermissionError,
                                    "search roots with the true hidden state are never allowed in training");
                    throw py::error_already_set();
                }
            }
        }
        EncodeCtx c = ctxFrom(ctx);
        py::object pyCtx = py::isinstance<py::dict>(ctx) ? py::object(ctx) : py::object(ctxDict(c));
        py::object encodeFn;
        std::vector<SearchRoot> rs;
        rs.reserve(roots.size());
        for (const py::handle& item : roots) {
            py::tuple t = py::reinterpret_borrow<py::tuple>(item);
            if (t.size() != 3) throw std::invalid_argument("roots: (Gen3Game, observer, decisions)");
            SearchRoot r;
            r.game = &t[0].cast<Gen3Game&>();
            r.decisions = t[2].cast<int>();
            py::handle o = t[1];
#if PKMN_HAVE_OBSERVER
            if (py::isinstance<ObsMemory>(o)) {
                r.obs = std::make_unique<CppNodeObs>(o.cast<const ObsMemory&>());
                rs.push_back(std::move(r));
                continue;
            }
#endif
            if (!encodeFn) encodeFn = py::module_::import("rl.encode").attr("battle");
            r.obs = std::make_unique<PyNodeObs>(py::reinterpret_borrow<py::object>(o), encodeFn, pyCtx);
            rs.push_back(std::move(r));
        }
        float rp[7];
        bool rl[7];
        for (int a = 0; a < 7; a++) {
            rp[a] = rootPriors[a];
            rl[a] = rootLegal[a];
        }
        BatchEvaluator ev = [&evaluator](const EncodedObs* obs, int n, float* pri, float* val) {
            py::object out = evaluator(batchDict(obs, n));
            py::tuple t = py::reinterpret_borrow<py::tuple>(out);
            if (t.size() != 2) throw std::runtime_error("evaluator must return (priors, values)");
            auto p = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(t[0]);
            auto v = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(t[1]);
            if (!p || !v || p.size() != static_cast<py::ssize_t>(n) * 7 || v.size() != n)
                throw std::runtime_error("evaluator: priors must be [B, 7] and values [B]");
            std::memcpy(pri, p.data(), sizeof(float) * n * 7);
            std::memcpy(val, v.data(), sizeof(float) * n);
            // one NaN value would poison every Q on its path (and the root's): fail loudly instead
            for (int i = 0; i < n; i++) {
                if (!std::isfinite(val[i])) throw std::runtime_error("evaluator returned a non-finite value");
                val[i] = std::min(1.0f, std::max(0.0f, val[i]));
            }
        };
        SearchStats st = runSearch(rs, c, rp, rl, rootValue, cfg, ev);
        py::dict out;
        out["visits"] = std::vector<double>(st.visits, st.visits + 7);
        out["q"] = std::vector<double>(st.q, st.q + 7);
        out["leaves"] = st.leaves;
        out["nodes"] = st.nodes;
        out["net_calls"] = st.netCalls;
        out["errors"] = st.errors;
        out["ms"] = st.msTotal;
        out["ms_eval"] = st.msEval;
        out["ms_cpp"] = st.msTotal - st.msEval;
        return out;
    }
};

}  // namespace

void bind_search(py::module_& m) {
#if !PKMN_HAVE_OBSERVER
    m.attr("HAS_CPP_OBSERVER") = false;
#else
    m.attr("HAS_CPP_OBSERVER") = true;
#endif
    py::class_<Searcher>(m, "Searcher",
                         "The batched PUCT loop of rl/search.py SearchBattler in C++ over prepared roots.")
        .def(py::init([](int nSims, int batch, float cPuct, float virtualLoss, int maxDecisions) {
                 Searcher s;
                 s.cfg.nSims = nSims;
                 s.cfg.batch = batch;
                 s.cfg.cPuct = cPuct;
                 s.cfg.virtualLoss = virtualLoss;
                 s.cfg.maxDecisions = maxDecisions;
                 return s;
             }),
             py::arg("n_sims") = 256, py::arg("batch") = 32, py::arg("c_puct") = 1.5f,
             py::arg("virtual_loss") = 1.0f, py::arg("max_decisions") = 300)
        .def("search", &Searcher::search, py::arg("roots"), py::arg("ctx"), py::arg("root_priors"),
             py::arg("root_legal"), py::arg("root_value"), py::arg("evaluator"),
             "roots: [(Gen3Game, ObsMemory | BattleObserver, decisions so far)], prepared (determinized, rebased, "
             "RNG set) and only cloned from. ctx: {'streak','battle','challenge','rents'} or EncodeCtx. "
             "evaluator(batch: dict of numpy arrays as rl.encode.collate stacks them) -> (priors [B,7], values [B]). "
             "-> {'visits', 'q', 'leaves', 'nodes', 'net_calls', 'errors', 'ms', 'ms_eval', 'ms_cpp'}")
        .def_property_readonly("n_sims", [](const Searcher& s) { return s.cfg.nSims; })
        .def_property_readonly("batch", [](const Searcher& s) { return s.cfg.batch; })
        .def_property_readonly("c_puct", [](const Searcher& s) { return s.cfg.cPuct; })
        .def_property_readonly("virtual_loss", [](const Searcher& s) { return s.cfg.virtualLoss; })
        .def_property_readonly("max_decisions", [](const Searcher& s) { return s.cfg.maxDecisions; });
    m.def("has_choice", [](Gen3Game& g) { return hasChoice(g); }, py::arg("game"),
          "At an ACTION decision: a usable move or a switch target exists (else FactoryEnv plays move 0).");
}
