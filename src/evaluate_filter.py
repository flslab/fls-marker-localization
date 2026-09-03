#!/usr/bin/env python3
"""
Offline evaluation of the Kalman filter on raw PnP tracker log data.

Usage:
    python evaluate_filter.py <path_to_tracker_json>

Reads the tracker JSON, runs a KF over every pose, and:
  1. Writes a new JSON file  (*_filtered.json) with both raw and filtered tvec.
  2. Generates a 3-panel comparison plot (filter_comparison.png).
"""

import argparse
import json
import os
import sys
import numpy as np

# ── Styling constants ────────────────────────────────────────────────
COLORS = {
    "raw":      "#FF3B3B",   # coral red
    "filtered": "#4E9D94",   # teal
    "grid":     "#2D2D2D",
    "bg":       "#FFFFFF",
    "panel":    "#FFFFFF",
    "text":     "#000000",
    "accent":   "#FFFFFF",
}


# ── Minimal KF (mirrors C++ PositionKalmanFilter) ────────────────────
class PositionKalmanFilter:
    def __init__(self, process_noise_std=0.5, measurement_noise_std=0.02):
        self.q_std = process_noise_std
        self.r_std = measurement_noise_std
        self._init = False
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.H = np.zeros((3, 6)); self.H[0,0] = self.H[1,1] = self.H[2,2] = 1.0
        self.R = np.eye(3) * measurement_noise_std**2
        self._t = 0.0

    def update(self, z, t):
        z = np.asarray(z, dtype=float)
        if not self._init:
            self.x[:3] = z; self._t = t; self._init = True
            return z.copy()
        dt = t - self._t
        if dt <= 0:
            return self.x[:3].copy()
        self._t = t

        F = np.eye(6); F[0,3] = F[1,4] = F[2,5] = dt
        q = self.q_std**2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        Q = np.zeros((6,6))
        for i in range(3):
            Q[i,i]       = q*dt4/4
            Q[i,i+3]     = q*dt3/2
            Q[i+3,i]     = q*dt3/2
            Q[i+3,i+3]   = q*dt2

        xp = F @ self.x
        Pp = F @ self.P @ F.T + Q
        y  = z - self.H @ xp
        S  = self.H @ Pp @ self.H.T + self.R
        K  = Pp @ self.H.T @ np.linalg.inv(S)
        self.x = xp + K @ y
        self.P = (np.eye(6) - K @ self.H) @ Pp
        return self.x[:3].copy()


# ── I/O helpers ──────────────────────────────────────────────────────

def load_tracker_log(path):
    with open(path) as f:
        data = json.load(f)
    return data


def run_filter(data, process_noise_std=0.5, measurement_noise_std=0.02):
    """Run the KF over every pose in `data`, modifying frames in-place."""
    kfs = {}   # per marker_id
    times_raw = []
    raw_positions = []
    filtered_positions = []

    for frame in data["frames"]:
        for pose in frame.get("poses", []):
            mid = pose["marker_id"]
            if mid not in kfs:
                kfs[mid] = PositionKalmanFilter(process_noise_std, measurement_noise_std)
            tvec = pose["tvec"]
            t = frame["time"]
            if t > 1e11:  # ms -> s
                t /= 1000.0
            filt = kfs[mid].update(tvec, t)
            pose["tvec_filtered"] = filt.tolist()

            times_raw.append(t)
            raw_positions.append(tvec)
            filtered_positions.append(filt.tolist())

    # Store filter config in the output
    data.setdefault("config", {})
    data["config"]["kalman_filter_enabled"] = True
    data["config"]["kf_process_noise"] = process_noise_std
    data["config"]["kf_measurement_noise"] = measurement_noise_std

    return (np.array(times_raw) if times_raw else np.array([]),
            np.array(raw_positions) if raw_positions else np.empty((0,3)),
            np.array(filtered_positions) if filtered_positions else np.empty((0,3)))


def compute_stats(raw, filtered):
    stats = {}
    labels = ("x", "y", "z")
    for i, label in enumerate(labels):
        raw_std = np.std(np.diff(raw[:, i])) if len(raw) > 1 else 0
        filt_std = np.std(np.diff(filtered[:, i])) if len(filtered) > 1 else 0
        reduction = (1 - filt_std / raw_std) * 100 if raw_std > 0 else 0
        stats[label] = {
            "raw_jitter_std": raw_std,
            "filtered_jitter_std": filt_std,
            "noise_reduction_pct": reduction,
        }
    return stats


def plot_comparison(times, raw, filtered, stats, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator
    except ImportError:
        print("matplotlib is required for plotting:  pip install matplotlib",
              file=sys.stderr)
        return

    t_plot = times - times[0]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor(COLORS["bg"])

    labels = ("X  (right)", "Y  (down)", "Z  (forward)")
    units = "m"

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.set_facecolor(COLORS["panel"])
        ax.plot(t_plot, raw[:, i], color=COLORS["raw"], linewidth=0.7,
                alpha=0.65, label="Raw PnP")
        ax.plot(t_plot, filtered[:, i], color=COLORS["filtered"],
                linewidth=1.4, label="Kalman Filtered")

        s = stats[("x","y","z")[i]]
        ax.set_ylabel(f"{label}  [{units}]", color=COLORS["text"], fontsize=11)
        ax.tick_params(colors=COLORS["text"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(COLORS["accent"])

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="major", color=COLORS["grid"], lw=0.4, alpha=0.5)
        ax.grid(True, which="minor", color=COLORS["grid"], lw=0.2, alpha=0.25)

        ax.text(0.98, 0.92,
                f"jitter reduction: {s['noise_reduction_pct']:.1f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=COLORS["filtered"],
                bbox=dict(facecolor=COLORS["accent"], edgecolor="none",
                          alpha=0.6, pad=3))
        if i == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.5,
                      facecolor=COLORS["accent"], edgecolor="none",
                      labelcolor=COLORS["text"])

    axes[-1].set_xlabel("Time  [s]", color=COLORS["text"], fontsize=11)
    fig.suptitle("PnP Position — Raw vs Kalman Filtered",
                 color=COLORS["text"], fontsize=14, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="Path to tracker JSON log file")
    ap.add_argument("--process-noise", type=float, default=0.5,
                    help="Process noise std (m/s²), default 0.5")
    ap.add_argument("--measurement-noise", type=float, default=0.02,
                    help="Measurement noise std (m), default 0.02")
    ap.add_argument("-o", "--output", default=None,
                    help="Output JSON path (default: <log>_filtered.json)")
    ap.add_argument("--plot", default=None,
                    help="Output plot path (default: <log_dir>/filter_comparison.png)")
    args = ap.parse_args()

    # ── Load ──────────────────────────────────────────────────────────
    data = load_tracker_log(args.log)

    # ── Filter ────────────────────────────────────────────────────────
    times, raw, filtered = run_filter(
        data,
        process_noise_std=args.process_noise,
        measurement_noise_std=args.measurement_noise,
    )

    n = len(times)
    if n == 0:
        print("No frames with pose data found — nothing to filter.",
              file=sys.stderr)
        sys.exit(1)

    duration = times[-1] - times[0]
    rate = (n - 1) / duration if duration > 0 else 0
    print(f"Filtered {n} pose samples  ({duration:.2f}s, ~{rate:.0f} Hz)")

    # ── Statistics ────────────────────────────────────────────────────
    stats = compute_stats(raw, filtered)
    print("\nPer-axis frame-to-frame jitter (std of Δ):")
    for axis in ("x", "y", "z"):
        s = stats[axis]
        print(f"  {axis}: raw={s['raw_jitter_std']*1000:.3f} mm  "
              f"filtered={s['filtered_jitter_std']*1000:.3f} mm  "
              f"reduction={s['noise_reduction_pct']:.1f}%")

    # ── Write filtered JSON ───────────────────────────────────────────
    if args.output:
        out_json = args.output
    else:
        base, ext = os.path.splitext(args.log)
        out_json = base + "_filtered" + ext

    with open(out_json, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\nFiltered JSON saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────
    if n >= 2:
        plot_path = args.plot or os.path.join(
            os.path.dirname(args.log), "filter_comparison.png")
        plot_comparison(times, raw, filtered, stats, plot_path)
    else:
        print("Not enough data points for a meaningful plot (need ≥ 2).")


if __name__ == "__main__":
    main()
