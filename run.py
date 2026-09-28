#!/usr/bin/env python3
"""Command line entry point: python run.py --time 35 --case both."""
import argparse
import json
from pathlib import Path

import numpy as np

from active_subspace import estimate_active_subspace


def plot_result(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.array(result["eigenvalues"])
    count = result["selected_dimension"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].semilogy(np.arange(1, 8), np.maximum(values, values[0]*1e-16), "o-")
    if "bootstrap" in result:
        limits = np.array(result["bootstrap"]["eigenvalue_percentiles_2_5_97_5"])
        axes[0].fill_between(np.arange(1, 8), np.maximum(limits[0], values[0]*1e-16),
                             np.maximum(limits[1], values[0]*1e-16), alpha=0.2)
    axes[0].set(xlabel="Direction index", ylabel="Eigenvalue (K²)", title="Gradient energy spectrum")
    axes[0].grid(alpha=0.25)
    loadings = np.array(result["active_directions"])
    im = axes[1].imshow(loadings, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axes[1].set_xticks(range(count), [f"w{i+1}" for i in range(count)])
    axes[1].set_yticks(range(7), result["parameters"])
    axes[1].set_title("Directions in standardized inputs")
    for i in range(7):
        for j in range(count):
            axes[1].text(j, i, f"{loadings[i,j]:.3f}", ha="center", va="center",
                         color="white" if abs(loadings[i,j]) > .65 else "black")
    fig.colorbar(im, ax=axes[1], label="Loading")
    fig.suptitle(f"Case {result['case']}, t = {result['time_seconds']:g} s, X = {result['position_X']:g}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", type=float, nargs="+", help="One or more times in seconds; prompts if omitted.")
    parser.add_argument("--case", choices=["1", "2", "both"], default="both")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=["lhs", "mc"], default="lhs")
    parser.add_argument("--position", type=float, default=0.0, help="X=x/b from the tip (0=tip, 1=wall).")
    parser.add_argument("--terms", type=int, default=100, help="Minimum series terms.")
    parser.add_argument("--fixed-terms", action="store_true", help="Disable early-time term adaptation.")
    parser.add_argument("--energy", type=float, default=0.99)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        times = args.time if args.time is not None else [float(input("Time t in seconds: "))]
        cases = [1, 2] if args.case == "both" else [int(args.case)]
        for case in cases:
            for time in times:
                print(f"Computing Case {case} at t={time:g} s ({args.samples} samples)...", flush=True)
                result = estimate_active_subspace(
                    time, case, samples=args.samples, seed=args.seed, sampling=args.sampling,
                    position=args.position, terms=args.terms, adaptive=not args.fixed_terms,
                    energy=args.energy, bootstrap=args.bootstrap)
                n = result["selected_dimension"]
                directions = np.array(result["active_directions"])
                print(f"  {n} direction(s) capture {result['cumulative_energy'][n-1]:.3%} of gradient energy.")
                print("  Eigenvalues:", np.array2string(np.array(result["eigenvalues"]), precision=5))
                print("  Input     " + " ".join(f"{'w'+str(j+1):>10}" for j in range(n)))
                for name, row in zip(result["parameters"], directions):
                    print(f"  {name:<10}" + " ".join(f"{v:10.6f}" for v in row))
                print("  Activity ranking:", " > ".join(result["parameter_ranking"]))
                args.output.mkdir(parents=True, exist_ok=True)
                stem = f"case{case}_t{time}_X{args.position}"
                path = args.output / (stem + ".json")
                path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                if args.plot:
                    plot_result(result, args.output / (stem + ".png"))
                print(f"  Saved {path}", flush=True)
    except (ValueError, ImportError, FloatingPointError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
