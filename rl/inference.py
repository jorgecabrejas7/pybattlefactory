"""GPU inference server for the battler network: one process holds FactoryNet on CUDA and serves N worker
processes through shared memory, batching their requests into one forward pass.

    with InferenceServer("runs/ppo_joint_v3/latest.pt", n_clients=18) as server:   # before forking the workers
        ...fork workers; worker p:  client = server.client(p)
        priors, values = client.evaluate(batch)     # batch: dict of numpy arrays, v3 battler layout

A request is a dict of numpy arrays with a leading batch dimension B (the keys of encode.battle):
    mon_ids int64 [B,6,19]  mon_num float32 [B,6,MON_NUM]  move_num float32 [B,6,4,MOVE_NUM]
    ctx_ids int64 [B,2]     ctx_num float32 [B,CTX_NUM]    mask bool [B,7]    active int64 [B]
The answer is (priors float32 [B,7], values float32 [B]): exactly what rl.search.SearchBattler.evaluate computes
(battler_outputs below): the masked softmax of FactoryNet.battler's logits, and its value denormalized with the
checkpoint's battler value_norm and clipped to [0, 1].

Transport. Each client owns a shared-memory block (multiprocessing.shared_memory) holding a small header
(state, sequence number, batch size), its input arrays (up to max_batch samples), its outputs and an error
message. The client writes its arrays, sets state = PENDING and releases one shared "request" semaphore; it then
waits on its own "response" semaphore. Nothing is pickled. The server scans the headers, waits up to deadline_ms
for more requests (or until every recently active client has one pending, or max_total samples are gathered),
runs one forward pass under torch.inference_mode (fp32, TF32 off; optionally bf16 autocast, not recommended:
~0.14 max prior error), replayed as a CUDA graph captured for the next bucket of batch sizes (the eager forward
is bound by ~100 kernel launches: 3-4 ms on a busy CPU, against 0.2-0.8 ms replayed), scatters the results and
releases each client's semaphore.

Benchmark: python -m rl.inference [--workers 1,8,18 --sizes 32,256].

Robustness. The owner (the process that built the InferenceServer) unlinks the shared memory on close(), on
__exit__ and at exit. A worker that dies is harmless: its request, if any, is answered to nobody, and the
server never waits for a particular client (only up to deadline_ms). A client waiting for an answer checks that
the server process is alive every `poll` seconds and raises ServerDied if not, TimeoutError after `timeout`
seconds. The server exits when its owner dies.

make_evaluator(...) returns the generic batch evaluator (callable batch_dict -> (priors, values)) used by the
Python search (obs_evaluator) and by the C++ Searcher: a client of this server, or a local network on any device.
"""

import atexit
import os
import time
import traceback

import numpy as np
import torch

from . import encode

KEYS = ("mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "mask", "active")
N_ACTIONS = 7
IDLE, PENDING, WORKING, DONE, ERROR = 0, 1, 2, 3, 4
_HDR = 8                        # int64 header: state, seq, n, answered seq, (reserved)
_ERR = 1024                     # bytes of error message
_CTL = 8                        # float64 control block: closed flag, batches, samples, forward seconds, ...


FACTORYNET_ALGOS = ("ppo", "alphazero")      # checkpoints whose network is FactoryNet


class ServerDied(RuntimeError):
    pass


# ---- the network's battler outputs (shared by the CPU path and the server) ---------------------------------------

def value_affine(policy):
    """(mean, std) of the checkpoint's battler value normalization, as SearchBattler uses it."""
    norm = (getattr(policy, "value_norm", None) or {}).get("battler")
    return (norm["mean"], max(norm["var"], 1e-4) ** 0.5) if norm else (0.0, 1.0)


def battler_outputs(net, x, v_mean, v_std):
    """Batched tensors -> (priors [B,7], values [B] in [0, 1]) as float32 tensors (SearchBattler.evaluate)."""
    logits, v = net.battler(x)
    pri = torch.softmax(logits.float(), -1)
    val = (v.float() * v_std + v_mean).clamp(0.0, 1.0)
    return pri, val


def layout(encode_version=None):
    """Per-sample shapes and dtypes of a battler request for an encoding version (default: the current one)."""
    old = encode.VERSION
    if encode_version is not None and encode_version != old:
        encode.set_version(encode_version)
    try:
        return {"mon_ids": ((6, encode.MON_IDS), np.int64), "mon_num": ((6, encode.MON_NUM), np.float32),
                "move_num": ((6, 4, encode.MOVE_NUM), np.float32), "ctx_ids": ((encode.CTX_IDS,), np.int64),
                "ctx_num": ((encode.CTX_NUM,), np.float32), "mask": ((N_ACTIONS,), np.bool_),
                "active": ((), np.int64)}
    finally:
        if encode.VERSION != old:
            encode.set_version(old)


def stack_obs(obs_list):
    """encode.battle observations -> one batch dict (the request format)."""
    return {k: np.stack([o[k] for o in obs_list]) for k in KEYS}


def _autocast(device, precision):
    if precision == "fp32":
        return torch.autocast(device.type, enabled=False)
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return torch.autocast(device.type, dtype=dt, cache_enabled=False)


def _buckets(max_total):
    """CUDA-graph batch sizes: 1, 2, 4, ... 64, then steps of 64 up to 512, of 256 up to max_total."""
    b, out = 1, []
    while b < 64:
        out.append(b)
        b *= 2
    out += list(range(64, 512, 64)) + list(range(512, max_total, 256)) + [max_total]
    return sorted({x for x in out if x <= max_total})


class LocalEvaluator:
    """The battler network in this process (any device): the same callable interface as InferenceClient."""

    def __init__(self, policy, precision="fp32"):
        if getattr(policy, "algo", "ppo") not in FACTORYNET_ALGOS:
            raise ValueError("the battler evaluator needs a FactoryNet (PPO or AlphaZero) checkpoint")
        self.policy, self.net, self.device = policy, policy.net, policy.device
        self.v_mean, self.v_std = value_affine(policy)
        self.precision = precision

    def evaluate(self, batch):
        with torch.inference_mode(), _autocast(self.device, self.precision):
            x = {k: torch.from_numpy(np.ascontiguousarray(batch[k])).to(self.device) for k in KEYS}
            pri, val = battler_outputs(self.net, x, self.v_mean, self.v_std)
        return pri.cpu().numpy(), val.cpu().numpy()

    __call__ = evaluate


def make_evaluator(source, device="cpu", precision="fp32"):
    """A batch evaluator: callable(batch_dict) -> (priors float32 [B,7], values float32 [B]).
    source: an InferenceClient (the GPU server), an InferenceServer (its client 0: single-process use), a
    rl.policy.Policy (local network on its device) or a checkpoint path (loaded on `device`)."""
    if isinstance(source, InferenceClient):
        return source.evaluate
    if isinstance(source, InferenceServer):
        return source.client(0).evaluate
    if isinstance(source, (str, os.PathLike)):
        from .policy import Policy
        source = Policy(str(source), torch.device(device))
    return LocalEvaluator(source, precision).evaluate


def obs_evaluator(evaluator):
    """Adapt a batch evaluator to SearchBattler.evaluate's signature: list of encode.battle observations ->
    (priors, values)."""
    def evaluate(obs_list):
        return evaluator(stack_obs(obs_list))
    return evaluate


# ---- shared-memory slots -----------------------------------------------------------------------------------------

class _Slot:
    """Views on one client's shared-memory block."""

    def __init__(self, buf, lay, max_batch):
        off = 0

        def take(shape, dtype):
            nonlocal off
            dtype = np.dtype(dtype)
            off = (off + 63) & ~63
            n = int(np.prod(shape)) * dtype.itemsize
            a = np.ndarray(shape, dtype, buffer=buf, offset=off)
            off += n
            return a

        self.hdr = take((_HDR,), np.int64)
        self.x = {k: take((max_batch,) + shp, dt) for k, (shp, dt) in lay.items()}
        self.priors = take((max_batch, N_ACTIONS), np.float32)
        self.values = take((max_batch,), np.float32)
        self.err = take((_ERR,), np.uint8)
        self.nbytes = off

    def release(self):
        self.hdr = self.x = self.priors = self.values = self.err = None


def _slot_size(lay, max_batch):
    off = 0
    items = [((_HDR,), np.int64)] + [((max_batch,) + s, d) for s, d in lay.values()] + \
            [((max_batch, N_ACTIONS), np.float32), ((max_batch,), np.float32), ((_ERR,), np.uint8)]
    for shape, dt in items:
        off = (off + 63) & ~63
        off += int(np.prod(shape)) * np.dtype(dt).itemsize
    return off


def _pid_alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
        return state not in ("Z", "X")
    except FileNotFoundError:
        return False
    except OSError:                         # no procfs: signal 0
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


# ---- client -------------------------------------------------------------------------------------------------------

class InferenceClient:
    """One worker's connection (slot i). Build it with InferenceServer.client(i) in the owner, or in a forked
    child (the shared memory and semaphores are inherited). One request at a time per client."""

    def __init__(self, server, index):
        self.index, self.max_batch = index, server.max_batch
        self.timeout, self.poll = server.timeout, server.poll
        self._pid = server.pid
        self._shm, self._ctl_shm = server._shms[index], server._ctl_shm      # (keep the mappings alive)
        self._ctl = np.ndarray((_CTL,), np.float64, buffer=server._ctl_shm.buf)
        self._slot = _Slot(self._shm.buf, server.layout, server.max_batch)
        self._req, self._resp = server._req, server._resps[index]
        self._seq = int(self._slot.hdr[1])
        self.broken = None

    def evaluate(self, batch):
        """batch: dict of numpy arrays (KEYS, leading dim B) -> (priors float32 [B,7], values float32 [B])."""
        n = len(batch["active"])
        if n > self.max_batch:
            parts = [self.evaluate({k: batch[k][i:i + self.max_batch] for k in KEYS})
                     for i in range(0, n, self.max_batch)]
            return np.concatenate([p for p, _ in parts]), np.concatenate([v for _, v in parts])
        if self._ctl[0] and not self.broken:
            self.broken = "the server was closed"
        if self.broken:
            raise ServerDied(f"inference client {self.index} is unusable: {self.broken}")
        if n == 0:
            return np.zeros((0, N_ACTIONS), np.float32), np.zeros(0, np.float32)
        s = self._slot
        for k in KEYS:
            s.x[k][:n] = batch[k]
        self._seq += 1
        s.hdr[1] = self._seq
        s.hdr[2] = n
        s.hdr[0] = PENDING
        self._req.release()
        t_end = time.monotonic() + self.timeout
        while True:
            if self._resp.acquire(timeout=self.poll):
                if s.hdr[3] == self._seq and s.hdr[0] in (DONE, ERROR):
                    break
                continue                                    # a stale wake-up (an earlier, abandoned request)
            if self._ctl[0]:
                self.broken = "the server was closed"
                raise ServerDied(self.broken)
            if not _pid_alive(self._pid):
                self.broken = f"the inference server (pid {self._pid}) died"
                raise ServerDied(self.broken)
            if time.monotonic() > t_end:
                self.broken = f"no answer in {self.timeout} s"
                raise TimeoutError(f"inference client {self.index}: {self.broken}")
        if s.hdr[0] == ERROR:
            msg = bytes(s.err).split(b"\0", 1)[0].decode(errors="replace")
            s.hdr[0] = IDLE
            raise RuntimeError(f"inference server error: {msg}")
        pri, val = s.priors[:n].copy(), s.values[:n].copy()
        s.hdr[0] = IDLE
        return pri, val

    __call__ = evaluate


# ---- server (owner side) --------------------------------------------------------------------------------------------

class InferenceServer:
    """Owner of the server process and of the shared memory. Create it before forking the workers.

    ckpt         PPO checkpoint (rl.policy.Policy)
    n_clients    number of client slots (one per worker)
    max_batch    samples per request (larger requests are split by the client)
    max_total    samples per forward pass
    deadline_ms  how long the server waits for more requests after the first one (it stops waiting when every
                 client that sent a request in the last active_ms has one pending)
    precision    "fp32" (default; TF32 off) or "bf16" autocast
    timeout      seconds a client waits for an answer before raising TimeoutError
    cuda_graphs  replay captured CUDA graphs (batch padded to the next bucket) instead of the eager forward
    """

    def __init__(self, ckpt, n_clients, max_batch=1024, max_total=2048, deadline_ms=0.75, precision="fp32",
                 device="cuda", timeout=120.0, poll=0.5, start_timeout=300.0, threads=2, cuda_graphs=True,
                 active_ms=100.0):
        import multiprocessing as mp
        from multiprocessing import shared_memory
        if precision not in ("fp32", "bf16"):
            # (fp16 cannot hold FactoryNet.battler's -1e9 logit mask)
            raise ValueError(f"precision must be fp32 or bf16, not {precision!r}")
        if max_total < max_batch:
            raise ValueError("max_total must be >= max_batch")
        args = torch.load(ckpt, map_location="cpu")["args"]
        if args.get("algo", "ppo") not in FACTORYNET_ALGOS:
            raise ValueError("the inference server serves FactoryNet (PPO / AlphaZero) checkpoints")
        self.ckpt, self.n_clients, self.max_batch, self.max_total = str(ckpt), n_clients, max_batch, max_total
        self.deadline_ms, self.precision, self.device = deadline_ms, precision, str(device)
        self.timeout, self.poll = timeout, poll
        self.layout = layout(args.get("encode_version", 2))
        self._owner = os.getpid()
        self._closed = False
        ctx = mp.get_context("spawn")
        size = _slot_size(self.layout, max_batch)
        self._shms, self._ctl_shm = [], None
        try:
            self._ctl_shm = shared_memory.SharedMemory(create=True, size=_CTL * 8)
            np.ndarray((_CTL,), np.float64, buffer=self._ctl_shm.buf)[:] = 0
            for _ in range(n_clients):
                shm = shared_memory.SharedMemory(create=True, size=size)
                np.ndarray((_HDR,), np.int64, buffer=shm.buf)[:] = 0
                self._shms.append(shm)
            self._req = ctx.Semaphore(0)
            self._resps = [ctx.Semaphore(0) for _ in range(n_clients)]
            self._stop = ctx.Event()
            self._ctrl, ctrl_child = ctx.Pipe()                 # reload(): new weights between batches
            recv, send = ctx.Pipe(duplex=False)
            cfg = dict(ckpt=self.ckpt, max_batch=max_batch, max_total=max_total, deadline_ms=deadline_ms,
                       precision=precision, device=self.device, threads=threads, owner=self._owner,
                       cuda_graphs=cuda_graphs, active_ms=active_ms,
                       layout={k: (s, np.dtype(d).str) for k, (s, d) in self.layout.items()})
            self._proc = ctx.Process(target=_serve, name="inference-server", daemon=True,
                                     args=(cfg, [s.name for s in self._shms], self._ctl_shm.name, self._req,
                                           self._resps, self._stop, send, ctrl_child))
            self._proc.start()
            send.close()
            self.pid = self._proc.pid
            atexit.register(self.close)
            t_end = time.monotonic() + start_timeout
            while not recv.poll(0.2):
                if not self._proc.is_alive():
                    raise ServerDied(f"the inference server exited during start-up (code {self._proc.exitcode})")
                if time.monotonic() > t_end:
                    raise TimeoutError("the inference server did not start")
            status, info = recv.recv()
            recv.close()
            if status != "ok":
                raise RuntimeError(f"the inference server failed to start:\n{info}")
            self.info = info
        except BaseException:
            self.close()
            raise

    def reload(self, ckpt, timeout=120.0):
        """Load another checkpoint's weights and value normalization into the running server (same architecture
        and encoding). The weights are copied in place, so the captured CUDA graphs stay valid; the requests
        answered after this returns use the new network (it happens between two forward passes)."""
        if self._closed:
            raise ServerDied("the server was closed")
        self._ctrl.send(("reload", str(ckpt)))
        self._req.release()                                 # wake the server if it waits for requests
        if not self._ctrl.poll(timeout):
            raise TimeoutError("the inference server did not reload in time")
        status, info = self._ctrl.recv()
        if status != "ok":
            raise RuntimeError(f"the inference server could not reload {ckpt}:\n{info}")
        self.ckpt = str(ckpt)
        return info

    def client(self, i):
        if not 0 <= i < self.n_clients:
            raise IndexError(f"client slot {i} out of range 0..{self.n_clients - 1}")
        return InferenceClient(self, i)

    def stats(self):
        c = np.ndarray((_CTL,), np.float64, buffer=self._ctl_shm.buf)
        b, n = int(c[1]), int(c[2])
        return {"batches": b, "samples": n, "mean_batch": n / max(b, 1), "forward_ms": c[3] * 1000 / max(b, 1),
                "requests": int(c[4])}

    def alive(self):
        return getattr(self, "_proc", None) is not None and self._proc.is_alive()

    def close(self):
        """Stop the server and unlink the shared memory (owner only; idempotent)."""
        if self._closed or os.getpid() != self._owner:
            return
        self._closed = True
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
        proc = getattr(self, "_proc", None)
        if proc is not None and proc._popen is not None:
            self._stop.set()
            self._req.release()
            proc.join(10)
            if proc.is_alive():
                proc.kill()
                proc.join(5)
        if self._ctl_shm is not None:
            np.ndarray((_CTL,), np.float64, buffer=self._ctl_shm.buf)[0] = 1.0     # clients: "closed"
        # unlink only: the mappings stay valid for clients living in this process (they hold the SharedMemory
        # objects and raise ServerDied from now on); the memory is released with the last mapping
        for shm in self._shms + ([self._ctl_shm] if self._ctl_shm is not None else []):
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---- server process ---------------------------------------------------------------------------------------------------

def _serve(cfg, shm_names, ctl_name, req, resps, stop, conn, ctrl=None):
    from multiprocessing import shared_memory
    shms, slots = [], []
    try:
        from .policy import Policy
        torch.set_num_threads(cfg["threads"])
        device = torch.device(cfg["device"])
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        policy = Policy(cfg["ckpt"], device)
        net = policy.net
        # (0-dim tensors, not Python floats: a CUDA graph reads them at replay, so reload() can change them)
        v_mean, v_std = (torch.tensor(x, dtype=torch.float32, device=device) for x in value_affine(policy))
        lay = {k: (tuple(s), np.dtype(d)) for k, (s, d) in cfg["layout"].items()}
        for name in shm_names + [ctl_name]:
            # (attaching registers the name with the resource tracker this process shares with the owner: a
            # no-op, the owner's unlink unregisters it)
            shms.append(shared_memory.SharedMemory(name=name))
        ctl = np.ndarray((_CTL,), np.float64, buffer=shms[-1].buf)
        slots = [_Slot(s.buf, lay, cfg["max_batch"]) for s in shms[:-1]]
        max_total, pin = cfg["max_total"], device.type == "cuda"
        tdt = {np.dtype(np.int64): torch.int64, np.dtype(np.float32): torch.float32, np.dtype(np.bool_): torch.bool}
        stage = {k: torch.empty((max_total,) + s, dtype=tdt[d], pin_memory=pin) for k, (s, d) in lay.items()}
        stage_np = {k: t.numpy() for k, t in stage.items()}
        out_p = torch.empty((max_total, N_ACTIONS), dtype=torch.float32, pin_memory=pin)
        out_v = torch.empty((max_total,), dtype=torch.float32, pin_memory=pin)
        out_p_np, out_v_np = out_p.numpy(), out_v.numpy()

        def run(x):
            with _autocast(device, cfg["precision"]):
                pri, val = battler_outputs(net, x, v_mean, v_std)
                if cfg["precision"] != "fp32":
                    pri = torch.nan_to_num(pri, nan=1.0 / N_ACTIONS)
            return pri, val

        # inputs start as zeros with a legal mask (rows past n in a padded CUDA-graph batch are computed and
        # dropped: every sample is independent of the others)
        stage_np["mask"][:] = True
        for k in KEYS:
            if k != "mask":
                stage_np[k][:] = 0
        graphs = {}
        use_graphs = device.type == "cuda" and cfg["cuda_graphs"]
        if use_graphs:
            # the eager forward is bound by kernel launches (~100 per pass, milliseconds on a busy CPU): capture
            # one CUDA graph per bucket of batch sizes on static input buffers, replay the smallest bucket >= n
            static = {k: stage[k].to(device) for k in KEYS}
            with torch.inference_mode():
                side = torch.cuda.Stream(device)
                side.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(side):
                    for b in (1, max_total):
                        run({k: static[k][:b] for k in KEYS})
                torch.cuda.current_stream(device).wait_stream(side)
                pool = None
                for b in reversed(_buckets(max_total)):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool):
                        outs = run({k: static[k][:b] for k in KEYS})
                    pool = g.pool()
                    graphs[b] = (g, outs)
            bucket_of = np.zeros(max_total + 1, np.int64)
            for b in sorted(graphs, reverse=True):
                bucket_of[:b + 1] = b

        def forward(n):
            with torch.inference_mode():
                if use_graphs:
                    for k in KEYS:
                        static[k][:n].copy_(stage[k][:n], non_blocking=True)
                    g, (pri, val) = graphs[int(bucket_of[n])]
                    g.replay()
                else:
                    pri, val = run({k: stage[k][:n].to(device, non_blocking=True) for k in KEYS})
                out_p[:n].copy_(pri[:n], non_blocking=True)
                out_v[:n].copy_(val[:n], non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        for n in sorted({min(b, max_total) for b in (1, 32, 256, 1024, max_total)}):      # warm-up
            forward(n)
        conn.send(("ok", {"device": str(device), "name": torch.cuda.get_device_name(device)
                          if device.type == "cuda" else "cpu", "pid": os.getpid()}))
        conn.close()
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
            conn.close()
        except Exception:
            pass
        return

    deadline = cfg["deadline_ms"] / 1000.0
    n_clients = len(slots)
    owner = cfg["owner"]
    last_check = time.monotonic()

    active_window = cfg["active_ms"] / 1000.0
    last_seen = np.full(n_clients, -1e18)

    def scan():
        return [i for i, s in enumerate(slots) if s.hdr[0] == PENDING]

    def reload(path):
        ck = torch.load(path, map_location=device)
        net.load_state_dict(ck["net"])                  # in place: the graphs' parameter buffers are kept
        policy.value_norm = ck.get("value_norm") or {}
        m, s = value_affine(policy)
        v_mean.fill_(m)
        v_std.fill_(s)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        while not stop.is_set():
            if ctrl is not None and ctrl.poll():
                cmd, arg = ctrl.recv()
                try:
                    if cmd != "reload":
                        raise ValueError(f"unknown command {cmd!r}")
                    reload(arg)
                    ctrl.send(("ok", {"ckpt": arg}))
                except Exception:
                    ctrl.send(("error", traceback.format_exc()))
            pending = scan()
            if not pending:
                if not req.acquire(timeout=0.2):
                    if os.getppid() != owner and not _pid_alive(owner):
                        break                                   # the owner is gone
                    continue
                pending = scan()
                if not pending:
                    continue
            total = sum(int(slots[i].hdr[2]) for i in pending)
            now = time.monotonic()
            last_seen[pending] = now
            # wait for the other clients seen recently (not for idle or dead ones), at most `deadline`
            active = int((last_seen > now - active_window).sum())
            t_end = now + deadline
            while total < max_total and len(pending) < active:
                rem = t_end - time.monotonic()
                if rem <= 0 or not req.acquire(timeout=rem):
                    break
                pending = scan()
                total = sum(int(slots[i].hdr[2]) for i in pending)
            last_seen[pending] = time.monotonic()
            if stop.is_set():
                break
            chosen, n = [], 0
            for i in pending:
                s = slots[i]
                m = int(s.hdr[2])
                if n + m > max_total:
                    continue
                s.hdr[0] = WORKING
                chosen.append((i, int(s.hdr[1]), n, m))
                for k in KEYS:
                    stage_np[k][n:n + m] = s.x[k][:m]
                n += m
            try:
                t0 = time.perf_counter()
                forward(n)
                ctl[3] += time.perf_counter() - t0
                ctl[1] += 1
                ctl[2] += n
                ctl[4] += len(chosen)
                for i, seq, o, m in chosen:
                    s = slots[i]
                    s.priors[:m] = out_p_np[o:o + m]
                    s.values[:m] = out_v_np[o:o + m]
                    s.hdr[3] = seq
                    s.hdr[0] = DONE
                    resps[i].release()
            except Exception:
                msg = traceback.format_exc().encode()[-(_ERR - 1):]
                for i, seq, o, m in chosen:
                    s = slots[i]
                    s.err[:] = 0
                    s.err[:len(msg)] = np.frombuffer(msg, np.uint8)
                    s.hdr[3] = seq
                    s.hdr[0] = ERROR
                    resps[i].release()
            now = time.monotonic()
            if now - last_check > 1.0:
                last_check = now
                if os.getppid() != owner and not _pid_alive(owner):
                    break
    finally:
        for s in slots:
            s.release()
        slots.clear()
        for shm in shms:
            try:
                shm.close()
            except BufferError:
                pass


# ---- benchmark ----------------------------------------------------------------------------------------------------

def _bench_client(server, slot, size, seconds, seed, q):
    c = server.client(slot)
    rng = np.random.default_rng(seed)
    batch = {}
    for k, (shp, dt) in server.layout.items():
        if k == "mask":
            batch[k] = rng.random((size,) + shp) < 0.7
            batch[k][:, 0] = True
        elif k == "active":
            batch[k] = rng.integers(0, 3, size)
        elif np.dtype(dt) == np.int64:
            batch[k] = rng.integers(0, 5, (size,) + shp)
        else:
            batch[k] = rng.random((size,) + shp).astype(np.float32)
    c.evaluate(batch)
    lat = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t = time.perf_counter()
        c.evaluate(batch)
        lat.append(time.perf_counter() - t)
    q.put(lat)


def bench(ckpt, workers=(1, 8, 18), sizes=(32, 256), seconds=3.0, **server_kw):
    """Closed-loop clients (each sends a request of `size` samples, waits, repeats) -> rows of throughput and
    latency."""
    import multiprocessing as mp
    rows = []
    with InferenceServer(ckpt, n_clients=max(workers), **server_kw) as server:
        ctx = mp.get_context("fork")
        for size in sizes:
            for w in workers:
                q = ctx.Queue()
                s0 = server.stats()
                procs = [ctx.Process(target=_bench_client, args=(server, i, size, seconds, i, q)) for i in range(w)]
                t0 = time.perf_counter()
                for p in procs:
                    p.start()
                lats = [q.get(timeout=seconds + 120) for _ in procs]
                wall = time.perf_counter() - t0
                for p in procs:
                    p.join(10)
                s1 = server.stats()
                lat = np.concatenate([np.asarray(x) for x in lats]) * 1000
                n_req = len(lat)
                b = s1["batches"] - s0["batches"]
                rows.append({"workers": w, "size": size, "requests": n_req,
                             "samples_per_s": n_req * size / seconds,
                             "lat_mean_ms": float(lat.mean()), "lat_p50_ms": float(np.median(lat)),
                             "lat_p99_ms": float(np.percentile(lat, 99)),
                             "mean_forward_batch": (s1["samples"] - s0["samples"]) / max(b, 1),
                             "forward_ms": (s1["forward_ms"] * s1["batches"] - s0["forward_ms"] * s0["batches"])
                             / max(b, 1), "wall_s": wall})
                r = rows[-1]
                print(f"workers {w:2d} size {size:4d}: {r['samples_per_s']:9.0f} samples/s  latency mean "
                      f"{r['lat_mean_ms']:6.2f} p50 {r['lat_p50_ms']:6.2f} p99 {r['lat_p99_ms']:6.2f} ms  "
                      f"forward batch {r['mean_forward_batch']:6.0f} ({r['forward_ms']:.2f} ms)", flush=True)
    return rows


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="benchmark the GPU inference server")
    ap.add_argument("--ckpt", default="runs/ppo_joint_v3/latest.pt")
    ap.add_argument("--workers", default="1,8,18")
    ap.add_argument("--sizes", default="32,256")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--deadline-ms", type=float, default=0.75)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = bench(a.ckpt, [int(x) for x in a.workers.split(",")], [int(x) for x in a.sizes.split(",")], a.seconds,
                deadline_ms=a.deadline_ms, precision=a.precision)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
