from pathlib import Path

"""
Sample from an initial distribution, apply a sequence of transformations,
and estimate the resulting PDF using a density-normalized histogram.

Examples
--------
    python transform_distribution.py
    python transform_distribution.py --distribution normal
    python transform_distribution.py --distribution uniform --low -1 --high 1
    python transform_distribution.py --samples 1000000 --bins 300

Edit `build_transform_pipeline()` to define the transformations.
"""

import argparse
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
Transform = Callable[[Array], Array]


def f1(x: np.ndarray) -> np.ndarray:

    return x * np.abs(x)



def build_transform_pipeline() -> list[tuple[str, Transform]]:
    """
    Define the transformations applied to the samples.

    Each tuple contains:
        ("name shown in the plot", transformation function)

    The functions are applied sequentially:

        x_0 ~ initial distribution
        x_1 = f_1(x_0)
        x_2 = f_2(x_1)
        ...
        y   = f_n(...f_2(f_1(x_0)))

    Replace, remove, or add transformations here.
    """
    return [
        ("f", f1)
        # ("tanh", np.tanh),
        # ("square", lambda x: x**2),
        # ("affine: 2x + 1", lambda x: 2.0 * x + 1.0),
        # ("sin", np.sin),
        # ("clip to [-0.8, 0.8]", lambda x: np.clip(x, -0.8, 0.8)),
        # ("sign(x) * sqrt(|x|)", lambda x: np.sign(x) * np.sqrt(np.abs(x))),
    ]


def sample_initial_distribution(
    rng: np.random.Generator,
    distribution: str,
    n_samples: int,
    mean: float,
    std: float,
    low: float,
    high: float,
) -> Array:
    """Draw samples from the selected initial distribution."""
    if distribution == "normal":
        if std <= 0:
            raise ValueError("--std must be positive.")
        return rng.normal(loc=mean, scale=std, size=n_samples)

    if distribution == "uniform":
        if high <= low:
            raise ValueError("--high must be larger than --low.")
        return rng.uniform(low=low, high=high, size=n_samples)

    raise ValueError(f"Unknown distribution: {distribution!r}")


def apply_pipeline(
    samples: Array,
    pipeline: Sequence[tuple[str, Transform]],
) -> tuple[Array, list[tuple[str, Array]]]:
    """
    Apply all transformations and retain intermediate sample arrays.

    Non-finite values are removed after every step. This is useful for
    transformations such as log(x), but a warning is printed.
    """
    current = np.asarray(samples, dtype=np.float64)
    intermediates: list[tuple[str, Array]] = []

    for name, transform in pipeline:
        with np.errstate(all="ignore"):
            transformed = np.asarray(transform(current), dtype=np.float64)

        if transformed.shape != current.shape:
            raise ValueError(
                f"Transformation {name!r} changed the shape from "
                f"{current.shape} to {transformed.shape}. "
                "Each transform must operate element-wise and preserve shape."
            )

        finite_mask = np.isfinite(transformed)
        removed = int((~finite_mask).sum())
        if removed:
            print(
                f"Warning: transformation {name!r} produced {removed:,} "
                "non-finite values; they were removed."
            )

        current = transformed[finite_mask]
        if current.size == 0:
            raise ValueError(
                f"Transformation {name!r} removed all samples."
            )

        intermediates.append((name, current.copy()))

    return current, intermediates


def robust_plot_range(samples: Array, tail_fraction: float) -> tuple[float, float]:
    """
    Choose a plotting range that ignores an optional fraction of extreme tails.

    tail_fraction=0.001 discards 0.05% from each side for plotting only.
    The samples themselves are not modified.
    """
    if not 0.0 <= tail_fraction < 1.0:
        raise ValueError("--trim-tails must lie in [0, 1).")

    if tail_fraction == 0.0:
        left, right = float(np.min(samples)), float(np.max(samples))
    else:
        q = tail_fraction / 2.0
        left, right = np.quantile(samples, [q, 1.0 - q])

    if np.isclose(left, right):
        padding = max(abs(left) * 0.05, 1e-6)
        left -= padding
        right += padding

    return float(left), float(right)


def plot_distributions(
    initial_samples: Array,
    transformed_samples: Array,
    pipeline: Sequence[tuple[str, Transform]],
    distribution_description: str,
    bins: int,
    trim_tails: float,
    show_intermediate: bool,
    output: str | None,
) -> None:
    """Plot density-normalized histograms of the sampled distributions."""
    rows = 2 if show_intermediate and pipeline else 1
    fig, axes = plt.subplots(rows, 1, figsize=(10, 5.5 * rows), squeeze=False)
    main_ax = axes[0, 0]

    initial_range = robust_plot_range(initial_samples, trim_tails)
    transformed_range = robust_plot_range(transformed_samples, trim_tails)

    main_ax.hist(
        initial_samples,
        bins=bins,
        range=initial_range,
        density=True,
        alpha=0.55,
        label=f"Initial: {distribution_description}",
    )
    main_ax.hist(
        transformed_samples,
        bins=bins,
        range=transformed_range,
        density=True,
        alpha=0.55,
        label="Final transformed distribution",
    )

    transform_names = " → ".join(name for name, _ in pipeline) or "identity"
    main_ax.set_title(f"Distribution transformation: {transform_names}")
    main_ax.set_xlabel("Value")
    main_ax.set_ylabel("Estimated probability density")
    main_ax.grid(alpha=0.25)
    main_ax.legend()

    if rows == 2:
        intermediate_ax = axes[1, 0]
        current = initial_samples

        intermediate_ax.hist(
            current,
            bins=bins,
            range=robust_plot_range(current, trim_tails),
            density=True,
            histtype="step",
            linewidth=1.6,
            label="Initial",
        )

        for name, transform in pipeline:
            with np.errstate(all="ignore"):
                current = np.asarray(transform(current), dtype=np.float64)
            current = current[np.isfinite(current)]

            intermediate_ax.hist(
                current,
                bins=bins,
                range=robust_plot_range(current, trim_tails),
                density=True,
                histtype="step",
                linewidth=1.6,
                label=name,
            )

        intermediate_ax.set_title("Intermediate distributions")
        intermediate_ax.set_xlabel("Value")
        intermediate_ax.set_ylabel("Estimated probability density")
        intermediate_ax.grid(alpha=0.25)
        intermediate_ax.legend()

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {output}")

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample an initial normal or uniform distribution, apply a "
            "transformation pipeline, and plot the resulting empirical PDF."
        )
    )
    parser.add_argument(
        "--distribution",
        choices=("normal", "uniform"),
        default="uniform",
    )
    parser.add_argument("--samples", type=int, default=300_000)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)

    parser.add_argument("--low", type=float, default=-1.0)
    parser.add_argument("--high", type=float, default=1.0)

    parser.add_argument(
        "--trim-tails",
        type=float,
        default=0.001,
        help=(
            "Fraction of extreme samples excluded from the plotting range. "
            "For example, 0.001 trims 0.05%% from each side. "
            "Use 0 to show the full sampled range."
        ),
    )
    parser.add_argument(
        "--show-intermediate",
        action="store_true",
        help="Also plot the distribution after each transformation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path, e.g. transformed_pdf.png",
    )

    args = parser.parse_args()

    if args.samples <= 0:
        parser.error("--samples must be positive.")
    if args.bins <= 0:
        parser.error("--bins must be positive.")

    return args


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    initial_samples = sample_initial_distribution(
        rng=rng,
        distribution=args.distribution,
        n_samples=args.samples,
        mean=args.mean,
        std=args.std,
        low=args.low,
        high=args.high,
    )

    pipeline = build_transform_pipeline()
    transformed_samples, _ = apply_pipeline(initial_samples, pipeline)

    if args.distribution == "normal":
        description = f"N({args.mean:g}, {args.std:g}²)"
    else:
        description = f"U({args.low:g}, {args.high:g})"

    print(f"Initial samples:     {initial_samples.size:,}")
    print(f"Transformed samples: {transformed_samples.size:,}")
    print(f"Initial mean/std:    {initial_samples.mean():.5f} / {initial_samples.std():.5f}")
    print(
        f"Final mean/std:      "
        f"{transformed_samples.mean():.5f} / "
        f"{transformed_samples.std():.5f}"
    )

    plot_distributions(
        initial_samples=initial_samples,
        transformed_samples=transformed_samples,
        pipeline=pipeline,
        distribution_description=description,
        bins=args.bins,
        trim_tails=args.trim_tails,
        show_intermediate=args.show_intermediate,
        output=args.output,
    )


if __name__ == "__main__":
    main()

