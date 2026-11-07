"""Stimulus-response experiment against the LIF organoid simulator.

Stimulates one channel on alternate loop ticks and compares spike counts on that
channel in stimulated vs. unstimulated ticks. The SDK's web viewer opens in your
browser so you can watch the evoked activity live.

Run from the repo root:

    uv run python experiments/example1.py
    uv run python experiments/example1.py --seconds 60 --channel 35 --coupling 0
"""
from __future__ import annotations

import argparse
import webbrowser
from collections import Counter

import cl

TICKS_PER_SECOND = 20
STIM = cl.StimDesign(200, -1.5, 200, 1.5)   # biphasic, cathodic-first, 0.3 nC per phase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=30.0, help="Experiment duration (default 30)")
    parser.add_argument("--channel", type=int, default=27, help="Channel to stimulate (default 27)")
    parser.add_argument("--coupling", type=float, default=4.0,
                        help="LIF coupling_gain_mV; 0 makes only the stimulated channel respond (default 4.0)")
    parser.add_argument("--seed", type=int, default=42, help="Simulator random seed (default 42)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the viewer in a browser")
    parser.add_argument("--no-hold", action="store_true", help="Exit as soon as the experiment ends")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # .env already selects the LIF source; this call overrides its config for this run.
    if cl.is_simulator():
        cl.sim.set_simulator_data_source(
            "organoid_simulator.lif_data_source:make_lif_source",
            config={"random_seed": args.seed, "coupling_gain_mV": args.coupling},
        )

    stim_channels = cl.ChannelSet(args.channel)
    spikes_on_target = {"stim": 0, "no_stim": 0}
    ticks = {"stim": 0, "no_stim": 0}
    spikes_by_channel: Counter[int] = Counter()

    with cl.open() as neurons:
        # The viewer runs only in the simulator and only in real time
        # (CL_SDK_ACCELERATED_TIME disables it). The SDK prints its URL;
        # there is no public accessor, so read it from the server handle.
        server = getattr(neurons, "_websocket_server", None)
        viewer_url = server.web_url if server is not None else None
        if viewer_url and not args.no_browser:
            webbrowser.open(viewer_url)

        print(f"Stimulating channel {args.channel} on alternate ticks for {args.seconds:g} s ...")
        stimulated_last_tick = False
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND, stop_after_seconds=args.seconds):
            # Spikes in this tick follow the stim (or lack of one) sent last tick.
            condition = "stim" if stimulated_last_tick else "no_stim"
            ticks[condition] += 1
            for spike in tick.analysis.spikes:
                spikes_by_channel[spike.channel] += 1
                if spike.channel == args.channel:
                    spikes_on_target[condition] += 1

            stimulated_last_tick = tick.iteration % 2 == 0
            if stimulated_last_tick:
                neurons.stim(stim_channels, STIM)

            if tick.iteration % TICKS_PER_SECOND == 0 and tick.iteration > 0:
                print(f"  t={tick.iteration // TICKS_PER_SECOND:>3d}s  "
                      f"ch{args.channel} spikes: stim={spikes_on_target['stim']}  "
                      f"no_stim={spikes_on_target['no_stim']}")

        print("\nResults")
        for condition in ("stim", "no_stim"):
            rate = spikes_on_target[condition] / max(ticks[condition], 1)
            print(f"  {condition:>7}: {spikes_on_target[condition]:5d} spikes on ch{args.channel} "
                  f"over {ticks[condition]} ticks ({rate:.2f} per tick)")
        print("  Most active channels:", ", ".join(f"ch{ch}={n}" for ch, n in spikes_by_channel.most_common(8)))

        if viewer_url and not args.no_hold:
            print(f"\nViewer still running at {viewer_url}")
            input("Press Enter to close it and exit ...")


if __name__ == "__main__":
    main()
