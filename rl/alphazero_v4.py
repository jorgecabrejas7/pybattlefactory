"""alphazero_v4 (docs/RL_DECISIONS.md §22): alphazero_v3 + an opponent model learned from the agent's own experience,
used by the battler's search at evaluation / inference instead of the strict sampler.

    python -m rl.alphazero_v4 --name alphazero_v4 [--opponent-model-init runs/opp_model_v3/model.pt]
    tensorboard --logdir runs

Everything is alphazero_v3 (rl/alphazero_v3.py, run through its main() with the OpponentModelExt hooks): Gumbel
battler with 256 simulations and the factory_sets sampler in the training search, the planner's features, hints, W
refresh, --t-imitation 0.1, WSD. The v4 defaults are v3's launch settings (--team-eval runs/team_eval_v3/model.pt,
--t-imitation 0.1, --eval-every 3, --eval-search-every 15) with --iterations 1500. What v4 adds:

  1. Every self-play worker's FactoryEnv is watched by an rl.opponent_model.ExperienceCollector: from the views and
     the player's own battle memory only, it records the rental screens (6 Pokemon seen whole), every opponent
     Pokemon sent out (species, moves used, item named) and, when we take one at the swap screen, that one whole.
     The records come back with the self-play samples each iteration and are added to one ExperienceModel (a table
     of counts, rl/opponent_model.py) in the main process: learned ONLINE, deterministic (workers merged in order).
  2. The model's counts are saved in every checkpoint ("opponent_model") and in runs/<name>/opponent_model.pt;
     --opponent-model-init starts them from a file (the warm start collected by playing alphazero_v3,
     python -m rl.opponent_model_data collect).
  3. The evaluation search (--eval-search-every) draws the opponents from the model (opponent_prior "experience").
     rl.eval_rounds does the same by default for a checkpoint that carries a model.
  4. --opponent-prior experience (default factory_sets, as v3) would also use it in the training search; the
     workers then reload the model every --opponent-model-refresh iterations.

Checkpoints: args as v3 (version 3: same network, tactician and features, so every v3 tool reads them) plus
args["variant"] = "alphazero_v4", and "opponent_model" (ExperienceModel.state()).
"""

import os
import time

from . import alphazero_v3 as V3
from . import opponent_model as OM
from .gumbel import GumbelBattler

VARIANT = "alphazero_v4"
V4_DEFAULTS = {"iterations": 1500, "team_eval": "runs/team_eval_v3/model.pt", "t_imitation": 0.1, "eval_every": 3,
               "eval_search_every": 15}


def build_parser():
    p = V3.build_parser()
    p.description = __doc__.split("\n")[0]
    p.set_defaults(name=time.strftime("alphazero_v4_%Y%m%d_%H%M%S"), **V4_DEFAULTS)
    for act in p._actions:
        if act.dest == "opponent_prior":
            act.choices = ("factory_sets", "strict", "experience")
            act.help = "the TRAINING search's opponent sampler (factory_sets: v3; experience: the learned model)"
    p.add_argument("--opponent-model-init", default=None,
                   help="start the opponent model's counts from this file (rl.opponent_model_data collect) or a "
                        "checkpoint holding one; default: empty (the strict sampler until experience arrives)")
    p.add_argument("--opponent-model-cfg", default="{}",
                   help="JSON overrides of rl.opponent_model.DEFAULT_CFG (smoothing, use_hint...)")
    p.add_argument("--opponent-model-refresh", type=int, default=5,
                   help="--opponent-prior experience: iterations between reloads of the model in the workers")
    p.add_argument("--eval-opponent-prior", choices=("experience", "strict"), default="experience",
                   help="the evaluation search's opponent sampler")
    return p


class SelfPlayerV4(V3.SelfPlayerV3):
    """v3's self-play worker whose environment is watched by an ExperienceCollector; play() also returns the new
    records ("opp_obs"). With cfg opponent_prior "experience" the training search draws from the model file."""

    def __init__(self, cfg, seed, evaluator=None):
        exp = cfg["opponent_prior"] == "experience"
        super().__init__(dict(cfg, opponent_prior="strict") if exp else cfg, seed, evaluator)
        self.cfg = cfg
        if exp:
            self.battler = GumbelBattler(self.evaluator, cfg["sims"], cfg["dets"], cfg["search_batch"], seed=seed,
                                         max_considered=cfg["max_considered"], c_visit=cfg["c_visit"],
                                         c_scale=cfg["c_scale"], rescale=bool(cfg["gumbel_rescale"]),
                                         non_root=cfg["non_root"], c_puct=cfg["c_puct"],
                                         opponent_prior="experience", opponent_model=cfg["opponent_model"])
        self.collector = OM.ExperienceCollector()
        self.collector.attach(self.env)

    def set_opponent_model(self, path):
        if self.battler.opponent_prior == "experience":
            self.battler.set_opponent_model(path)
        return path

    def play(self, n_decisions):
        out, st = super().play(n_decisions)
        out["opp_obs"] = dict(self.collector.drain())
        return out, st


class OpponentModelExt:
    """The alphazero_v3 main() hooks that make it alphazero_v4."""

    player = SelfPlayerV4

    def __init__(self):
        self.model = None
        self.path = None
        self.args = None

    def parse(self, argv):
        import json
        a = V3.finish_args(build_parser().parse_args(argv))
        a.opponent_model_cfg = json.loads(a.opponent_model_cfg) if isinstance(a.opponent_model_cfg, str) \
            else dict(a.opponent_model_cfg or {})
        self.args = a
        return a

    def config_extra(self, args):
        return {"variant": VARIANT}

    def load(self, ck, args, run_dir):
        """ck: the resumed v4 checkpoint (None for a new run): its counts; else --opponent-model-init or empty."""
        cfg = dict(args.opponent_model_cfg)
        if ck is not None and ck.get("opponent_model"):
            self.model = OM.ExperienceModel.from_state(ck["opponent_model"], cfg)
            src = "the checkpoint"
        elif args.opponent_model_init:
            self.model = OM.ExperienceModel.load(args.opponent_model_init, cfg)
            src = args.opponent_model_init
        else:
            self.model = OM.ExperienceModel(cfg)
            src = "empty"
        self.path = os.path.join(run_dir, "opponent_model.pt")
        self.model.save(self.path)
        print(f"opponent model ({src}): {self.model.summary()}", flush=True)

    def worker_cfg(self, cfg):
        return dict(cfg, opponent_model=self.path)

    def before_play(self, pool, it):
        if self.args.opponent_prior == "experience" and it > 0 and it % max(self.args.opponent_model_refresh, 1) == 0:
            self.model.save(self.path)
            pool.broadcast("call", ("set_opponent_model", (self.path,)))

    def after_play(self, results, tb, step, it):
        new = OM.merge([r.get("opp_obs") or {} for r, _ in results])
        self.model.add(new)
        s = self.model.summary()
        tb.add_scalar("opp_model/new_records", float(sum(new.values())), step)
        tb.add_scalar("opp_model/new_opponent_sightings", float(sum(v for k, v in new.items() if k[0] == OM.OPP)),
                      step)
        for k in ("records", "opponent_sightings", "opponent_full", "rentals", "configurations", "species"):
            tb.add_scalar(f"opp_model/{k}", float(s[k]), step)

    def ckpt_extra(self):
        return {"opponent_model": self.model.state()}

    def eval_search_kw(self):
        if self.args.eval_opponent_prior != "experience":
            return {"opponent_prior": "strict"}
        self.model.save(self.path)
        return {"opponent_prior": "experience", "opponent_model": self.path}


def main(argv=None):
    return V3.main(argv, ext=OpponentModelExt())


if __name__ == "__main__":
    main()
