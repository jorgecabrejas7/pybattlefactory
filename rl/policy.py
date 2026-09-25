"""Load a trained checkpoint (PPO or Rainbow) as a policy: policy.act(kind, x, greedy) -> actions in the
format decode_action expects (int, or (lead, pair) for rentals)."""

import numpy as np
import torch

from . import encode, ppo, rainbow
from .model import FactoryNet


class Policy:
    def __init__(self, path, device):
        ck = torch.load(path, map_location=device)
        a = ck["args"]
        self.algo = a.get("algo", "ppo")
        self.steps = ck.get("battler_steps", 0)
        self.device = device
        self.path = path
        self.value_norm = ck.get("value_norm") or {}      # per agent {"mean", "var"} (--value-norm 1), or None
        if self.algo == "rainbow":
            encode.set_version(a.get("encode_version", 2))
            self.net = rainbow.RainbowNet(a["atoms"], a["d_emb"], a["d"], a["layers"], a["heads"]).to(device)
            self.support = {"battle": torch.linspace(a["vmin_b"], a["vmax_b"], a["atoms"], device=device)}
            self.support["rental"] = self.support["swap"] = torch.linspace(a["vmin_t"], a["vmax_t"], a["atoms"],
                                                                           device=device)
        else:
            encode.set_version(a.get("encode_version", 2))     # v1/v2 checkpoints predate the versioned layout
            self.net = FactoryNet(a["d_emb"], a["d"], a["layers"], a["heads"], share=a.get("share", "all")).to(device)
        self.net.load_state_dict(ck["net"])
        self.net.eval()

    @torch.no_grad()
    def act(self, kind, x, greedy=True):
        if self.algo == "rainbow":
            # greedy: mean weights; "sampled": the noisy exploration policy used in training
            self.net.set_noise(not greedy)
            if not greedy:
                self.net.reset_noise()
            a, _ = rainbow.act(self.net, kind, x, self.support[kind])
            return np.array([rainbow.to_env_action(kind, v) for v in a], dtype=object) if kind == "rental" else a
        a, _, _ = ppo.act(self.net, kind, x, greedy=greedy)
        return a
