# Organoid simulator

The CL SDK's default simulator generates random spikes and ignores stimulation.
This package replaces it with a leaky integrate-and-fire network that responds
to `neurons.stim()`, so closed-loop code can be tested without hardware. The
experiment loop code stays the same.

| Source | Factory | Model | Speed |
|---|---|---|---|
| v1 LIF | `organoid_simulator.lif_data_source:make_lif_source` | Leaky integrate-and-fire, numpy | Faster than real time |

Copied from `organoid_reservoir_experiments/cl1_scripts/Benchmarking/organoid_simulator`.
Only the v1 LIF backend was brought across; the Brian2 v2 backend was not.

## Setup

Nothing to do. `uv sync` installs this repo as an editable package, which puts
`organoid_simulator` on `sys.path` for the interpreter, for the simulator
subprocess, and for notebooks.

The SDK builds the source in a subprocess and imports the factory by name, so
factories defined in notebooks or in `__main__` are rejected.

## It is already the default

The repo root `.env` sets `CL_SDK_DATA_SOURCE`, and `cl` calls `load_dotenv()`
when imported. Any script run from the repo root gets the LIF source with no
code change:

```python
import cl

with cl.open() as neurons:
    for tick in neurons.loop(ticks_per_second=50):
        neurons.stim(cl.ChannelSet(27), cl.StimDesign(200, -1.5, 200, 1.5))
        spikes = tick.analysis.spikes  # includes evoked spikes
```

`.env` pins `random_seed` only, leaving every other parameter at the values
documented below. To use different parameters, either edit
`CL_SDK_DATA_SOURCE_CONFIG` in `.env`, or override in code — an explicit call
takes precedence over the environment:

```python
if cl.is_simulator():  # set_simulator_data_source raises RuntimeError on a CL1
    cl.sim.set_simulator_data_source(
        "organoid_simulator.lif_data_source:make_lif_source",
        config={"random_seed": 42, "coupling_gain_mV": 4.0},  # JSON-serialisable
    )
```

Or per run, without touching the files:

```bash
CL_SDK_DATA_SOURCE_CONFIG='{"random_seed": 7, "coupling_gain_mV": 4.0}' \
CL_SDK_ACCELERATED_TIME=1 python your_experiment.py
```

## Common options

| Option | Default | Effect |
|---|---|---|
| `coupling_gain_mV` | `0.0` | Recurrent coupling strength. At 0, only the stimulated channel responds. |
| `connectivity` | small-world | `(64, 64)` matrix, `W[post, pre]`, as a nested list. The default is `small_world_connectivity()`; `grid_connectivity()` gives a 4-neighbour lattice. |
| `n_units` | `64` | Values other than 64 scatter neurons over the array and mix them into electrodes. |
| `plasticity`, `stdp` | `False` | Reward-modulated Hebbian learning or STDP on the coupling weights. Only available with 64 units. |
| `stim_artifact` | `False` | Adds a simple amplifier-saturation artifact. |
| `random_seed` | env or random | Seeds background drive and noise. Set to `42` in `.env`. |
| `separate_noise_streams` | `False` | With a fixed seed, makes background drive and recording noise independent of read batch boundaries. |

All parameters are documented in the `LIFDataSource` docstring.

## Model summary

- One LIF neuron per channel by default: tau 20 ms, threshold -50 mV, 2 ms refractory, plus Gaussian background drive.
- Only the cathodic phase of a pulse depolarises, because charge-balanced pulses would otherwise cancel.
- Frames are extracellular: a noise floor plus a stamped 80 uV spike waveform. Format is 25 kHz, 64-channel int16. Grounded channels (0, 7, 56, 63) and the reference channel (4) stay silent.
- No E/I cell types or conduction delays.

## Check

In a 10 s test with stimulation on channel 27 in alternate ticks, v1 with
`coupling_gain_mV=4.0` produced 248 spikes on channel 27 in post-stimulus ticks
and 17 in unstimulated ticks. The default random source produced 6 and 7.

## Limitations

Simulator results are not evidence about CL1 tissue. Report them separately from
device results. An accelerated CL SDK loop can still shift pulse timestamps
across runs; use recorded pulse times when checking complete-workflow
repeatability.
