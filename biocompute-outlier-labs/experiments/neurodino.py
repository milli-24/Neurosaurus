"""Does the simulated network's activity help time a jump over a variable-speed cactus?

Distance is spatially encoded on the array, so speed is only available from how the
input changes over time. All decoders are scored on trials they never saw, with the
same one-tick delay between input and readout.

Run from the repo root:

    CL_SDK_ACCELERATED_TIME=1 python experiments/neurodino.py --tps 50 --coupling 4
"""
from __future__ import annotations

import argparse
import json

import cl
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

STIM = cl.StimDesign(200, -1.5, 200, 1.5)
NON_SPIKING = {0, 4, 7, 56, 63}
LIVE = [c for c in range(64) if c not in NON_SPIKING]

# Must match game/index.html.
GRAVITY, JUMP_V, CLEAR_HEIGHT = 2600.0, 820.0, 44.0
GAP = 34 + 22                       # dino width + cactus width, px
GAME_TICK_S = 0.05
START_DIST = 300.0
SPEED_RANGE = (120.0, 480.0)

N_BINS = 16
BIN_PX = 16.0


def bin_channels(per_bin: int) -> list[list[int]]:
    """Electrodes stimulated for each distance bin, spread evenly across the array."""
    return [LIVE[i * 3:i * 3 + per_bin] for i in range(N_BINS)]


_root = np.sqrt(JUMP_V ** 2 - 2 * GRAVITY * CLEAR_HEIGHT)
T_UP, T_DOWN = (JUMP_V - _root) / GRAVITY, (JUMP_V + _root) / GRAVITY


def lead_window(speed: float) -> tuple[float, float]:
    """Earliest and latest time before contact at which a jump clears the cactus."""
    return T_UP, T_DOWN - GAP / speed


def distance_bin(dist: float) -> int | None:
    b = int(max(dist, 0.0) // BIN_PX)
    return b if b < N_BINS else None


def run_trials(n_trials: int, tps: int, coupling: float, seed: int, channels: list[list[int]]) -> dict:
    if cl.is_simulator():
        cl.sim.set_simulator_data_source(
            "organoid_simulator.lif_data_source:make_lif_source",
            config={"random_seed": seed, "coupling_gain_mV": coupling},
        )
    rng = np.random.default_rng(seed)
    speeds = rng.uniform(*SPEED_RANGE, n_trials)
    spikes, bins, dists, trial_ids = [], [], [], []
    trial, dist, prev_bin = 0, START_DIST, None

    with cl.open() as neurons:
        for tick in neurons.loop(ticks_per_second=tps):
            counts = np.zeros(64)
            for spike in tick.analysis.spikes:
                counts[spike.channel] += 1
            spikes.append(counts[LIVE])
            bins.append(-1 if prev_bin is None else prev_bin)
            dists.append(dist)
            trial_ids.append(trial)

            b = distance_bin(dist) if dist > 0 else None
            if b is not None:
                neurons.stim(cl.ChannelSet(*channels[b]), STIM)
            prev_bin = b

            dist -= speeds[trial] * GAME_TICK_S
            if dist <= 0:
                trial += 1
                dist, prev_bin = START_DIST, None
                if trial == n_trials:
                    break

    return {"spikes": np.array(spikes), "bins": np.array(bins), "dists": np.array(dists),
            "trials": np.array(trial_ids), "speeds": speeds}


def one_hot(bins: np.ndarray, trials: np.ndarray, history: int) -> np.ndarray:
    """One-hot of the bins stimulated in the last `history` ticks of the same trial."""
    x = np.zeros((len(bins), (N_BINS + 1) * history))
    for h in range(history):
        for i in range(len(bins)):
            j = i - h
            b = bins[j] if j >= 0 and trials[j] == trials[i] else -1
            x[i, h * (N_BINS + 1) + b + 1] = 1
    return x


def clear_rate(p: np.ndarray, data: dict, trial_set: np.ndarray, threshold: float) -> np.ndarray:
    """Per-trial outcome of jumping on the first tick where p crosses threshold."""
    outcomes = []
    for t in trial_set:
        idx = np.nonzero(data["trials"] == t)[0]
        above = idx[p[idx] >= threshold]
        if len(above) == 0:
            outcomes.append(0.0)
            continue
        speed = data["speeds"][t]
        lead = data["dists"][above[0]] / speed
        lo, hi = lead_window(speed)
        outcomes.append(float(lo <= lead <= hi))
    return np.array(outcomes)


def evaluate(x: np.ndarray, y: np.ndarray, data: dict, split: dict) -> tuple[np.ndarray, list]:
    """Held-out outcomes, plus the jump tick (or None) for each test trial."""
    mask = lambda s: np.isin(data["trials"], split[s])
    scaler = StandardScaler().fit(x[mask("train")])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(x[mask("train")]), y[mask("train")])
    p = model.predict_proba(scaler.transform(x))[:, 1]
    grid = np.linspace(0.05, 0.95, 37)
    best = max(grid, key=lambda th: clear_rate(p, data, split["val"], th).mean())
    jump_ticks = []
    for t in split["test"]:
        idx = np.nonzero(data["trials"] == t)[0]
        above = np.nonzero(p[idx] >= best)[0]
        jump_ticks.append(int(above[0]) if len(above) else None)
    return clear_rate(p, data, split["test"], best), jump_ticks


def speed_r2(x: np.ndarray, data: dict, split: dict) -> float:
    """Held-out R^2 for reading cactus speed from one tick of features."""
    mask = lambda s: np.isin(data["trials"], split[s]) & (data["bins"] >= 0)
    speed = data["speeds"][data["trials"]]
    train, test = mask("train"), mask("test")
    scaler = StandardScaler().fit(x[train])
    model = Ridge(alpha=1.0).fit(scaler.transform(x[train]), speed[train])
    return float(model.score(scaler.transform(x[test]), speed[test]))


def summary(name: str, outcomes: np.ndarray, rng: np.random.Generator) -> dict:
    boot = [rng.choice(outcomes, len(outcomes)).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:<34} {outcomes.mean():6.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    return {"name": name, "clear_rate": float(outcomes.mean()), "ci": [float(lo), float(hi)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--tps", type=int, default=50, help="Network ticks per second")
    parser.add_argument("--coupling", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Write results JSON here")
    parser.add_argument("--replay-out", default=None, help="Write held-out trials for the game here")
    parser.add_argument("--bin-channels", type=int, default=3, choices=(1, 2, 3), help="Electrodes stimulated per distance bin")
    parser.add_argument("--readout", default="all", choices=("all", "non-input"),
                        help="non-input drops every electrode used for stimulation from the decoder")
    args = parser.parse_args()

    channels = bin_channels(args.bin_channels)
    data = run_trials(args.trials, args.tps, args.coupling, args.seed, channels)
    rng = np.random.default_rng(args.seed + 1)

    input_set = {c for group in channels for c in group}
    keep = [i for i, c in enumerate(LIVE) if args.readout == "all" or c not in input_set]
    features = data["spikes"][:, keep]
    # Same spikes matched to the wrong moments: removes any link to the game, keeps the firing statistics.
    shuffled = features[np.random.default_rng(args.seed + 2).permutation(len(features))]

    speed = data["speeds"][data["trials"]]
    target = np.array([np.mean(lead_window(s)) for s in speed])
    y = (data["dists"] / speed <= target).astype(int)

    order = rng.permutation(args.trials)
    n_train, n_val = int(0.6 * args.trials), int(0.2 * args.trials)
    split = {"train": order[:n_train], "val": order[n_train:n_train + n_val], "test": order[n_train + n_val:]}

    print(f"\ntps={args.tps} coupling={args.coupling} seed={args.seed} bin_channels={args.bin_channels} "
          f"readout={args.readout} readout_electrodes={len(keep)} "
          f"ticks={len(y)} test_trials={len(split['test'])} spikes/tick={features.sum(1).mean():.2f}")
    results = []
    random_jump = []
    for t in split["test"]:
        idx = np.nonzero(data["trials"] == t)[0]
        lo, hi = lead_window(data["speeds"][t])
        random_jump.append(float(lo <= data["dists"][rng.choice(idx)] / data["speeds"][t] <= hi))
    random_out = np.array(random_jump)
    results.append(summary("random jump tick", random_out, rng))
    input_now = one_hot(data["bins"], data["trials"], 1)
    input_hist = one_hot(data["bins"], data["trials"], 6)
    now_out, now_jumps = evaluate(input_now, y, data, split)
    net_out, net_jumps = evaluate(features, y, data, split)
    shuf_out, _ = evaluate(shuffled, y, data, split)
    hist_out, hist_jumps = evaluate(input_hist, y, data, split)
    results.append(summary("shuffled network spikes", shuf_out, rng))
    results.append(summary("input now only (no memory)", now_out, rng))
    results.append(summary("network spikes this tick", net_out, rng))
    results.append(summary("input history, 6 ticks (silicon)", hist_out, rng))

    diffs = {}
    for name, other in (("input_now", now_out), ("shuffled", shuf_out), ("random", random_out)):
        diff = net_out - other
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        ci = [float(v) for v in np.percentile(boot, [2.5, 97.5])]
        diffs[name] = {"mean": float(diff.mean()), "ci": ci}
        print(f"  network minus {name}: {diff.mean():+.1%}   95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}]")

    r2 = {"input now": speed_r2(input_now, data, split),
          "network spikes": speed_r2(features, data, split),
          "shuffled spikes": speed_r2(shuffled, data, split),
          "input history": speed_r2(input_hist, data, split)}
    print("  speed R^2: " + "  ".join(f"{k}={v:.3f}" for k, v in r2.items()))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tps": args.tps, "coupling": args.coupling, "seed": args.seed, "trials": args.trials,
                       "bin_channels": args.bin_channels, "readout": args.readout, "readout_electrodes": len(keep),
                       "results": results, "network_minus": diffs, "speed_r2": r2}, f, indent=2)

    if args.replay_out:
        trials = []
        for i, t in enumerate(split["test"]):
            idx = np.nonzero(data["trials"] == t)[0]
            full = np.zeros((len(idx), 64), dtype=int)
            full[:, LIVE] = data["spikes"][idx].astype(int)
            trials.append({"speed": float(data["speeds"][t]), "start_dist": START_DIST, "spikes": full.tolist(),
                           "jump_tick": {"network": net_jumps[i], "input_now": now_jumps[i], "history": hist_jumps[i]}})
        with open(args.replay_out, "w") as f:
            json.dump({"tps": args.tps, "coupling": args.coupling, "seed": args.seed, "trials": trials}, f)


if __name__ == "__main__":
    main()
