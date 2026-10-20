import argparse
from pathlib import Path

import numpy as np
from obspy.signal.filter import bandpass
from scipy.signal import detrend, fftconvolve, hilbert


CACHE_FS = 4.0
ANALYSIS_START = 0.0
ANALYSIS_END = 4500.0
FILTER_CORNERS = 4
MINIMUM_SMOOTH_SECONDS = 25.0
SMOOTH_PERIOD_NUMBER = 10.0
GAUSSIAN_TRUNCATE = 4.0
GAP_EDGE_GUARD_PERIOD_NUMBER = 3.0
MINIMUM_SIM_SUPPORT_FRACTION = 0.65

CASE_NAMES = tuple(f"Case{i}" for i in range(1, 13))

# Keep the NPZ band order; downstream code can reorder by band_names.
BANDS = [
    ("global", 0.30, 0.80),
    ("HP3", 0.70, 0.80),
    ("HP2", 0.60, 0.70),
    ("HP1", 0.50, 0.60),
    ("MP2", 0.40, 0.50),
    ("MP1", 0.30, 0.40),
    ("LP", 0.20, 0.30),
]

# (Epicentral distance, event/station, Apollo cache, simulation cache).
RECORDS = [
    ("2.2°", "A14 LM→S14", "A14_LM_S14_minus5000_plus8000_DISP_4Hz.npz", "S22_local_radial_4Hz.npz"),
    ("2.4°", "A12 LM→S12", "A12_LM_S12_minus5000_plus8000_DISP_4Hz.npz", "S24_local_radial_4Hz.npz"),
    ("3.0°", "A15 LM→S15", "A15_LM_S15_minus5000_plus8000_DISP_4Hz.npz", "S30_local_radial_4Hz.npz"),
    ("3.8°", "A14 LM→S12", "A14_LM_S12_minus5000_plus8000_DISP_4Hz.npz", "S38_local_radial_4Hz.npz"),
    ("4.5°", "A13 S-IVB→S12", "A13_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz", "S45_local_radial_4Hz.npz"),
    ("5.1°", "A16 S-IVB→S12", "A16_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz", "S51_local_radial_4Hz.npz"),
    ("5.8°", "A14 S-IVB→S12", "A14_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz", "S58_local_radial_4Hz.npz"),
    ("6.1°", "A15 S-IVB→S14", "A15_SIVB_S14_minus5000_plus8000_DISP_4Hz.npz", "S61_local_radial_4Hz.npz"),
    ("9.1°", "A16 S-IVB→S14", "A16_SIVB_S14_minus5000_plus8000_DISP_4Hz.npz", "S91_local_tangential_4Hz.npz"),
    ("11.1°", "A17 S-IVB→S12", "A17_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz", "S111_local_radial_4Hz.npz"),
    ("11.7°", "A15 S-IVB→S12", "A15_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz", "S117_local_radial_4Hz.npz"),
    ("25.4°", "A17 LM→S15", "A17_LM_S15_minus5000_plus8000_DISP_4Hz.npz", "S254_local_radial_4Hz.npz"),
    ("28.0°", "A17 S-IVB→S16", "A17_SIVB_S16_minus5000_plus8000_DISP_4Hz.npz", "S280_local_radial_4Hz.npz"),
]


def load_cache(path):
    """Read the three required cache fields and reconstruct its time axis."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cache does not exist: {path}")
    with np.load(path, allow_pickle=False) as z:
        for key in ("data", "t0", "dt"):
            if key not in z.files:
                raise ValueError(f"Cache is missing {key!r}: {path}")
        data = np.asarray(z["data"], dtype=np.float64)
        t0 = float(z["t0"])
        dt = float(z["dt"])

    if data.ndim != 1 or len(data) < 100:
        raise ValueError(f"Cache data must have at least 100 samples: {path}")
    fs = 1.0 / dt
    if not np.isclose(fs, CACHE_FS, rtol=1e-7, atol=1e-10):
        raise ValueError(f"Expected {CACHE_FS:g} Hz, found {fs:.9f} Hz: {path}")
    time = t0 + np.arange(len(data), dtype=np.float64) * dt
    return time, data


def finite_runs(mask):
    """Yield half-open intervals of contiguous valid samples."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return zip(starts, ends)


def smooth_seconds_for_band(fmin, fmax):
    center_frequency = 0.5 * (fmin + fmax)
    return max(MINIMUM_SMOOTH_SECONDS, SMOOTH_PERIOD_NUMBER / center_frequency)


def gaussian_smooth(data, dt, fwhm_seconds):
    """Apply the source script's reflected-edge FFT Gaussian smoothing."""
    sigma_samples = max((fwhm_seconds / 2.355) / dt, 1.0)
    half_width = min(int(np.ceil(GAUSSIAN_TRUNCATE * sigma_samples)), len(data) - 1)
    x = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(data, (half_width, half_width), mode="reflect")
    smoothed = fftconvolve(padded, kernel, mode="same")
    smoothed = smoothed[half_width:half_width + len(data)]
    return np.maximum(smoothed, 0.0)


def process_segment(data, dt, fmin, fmax, smooth_seconds):
    """Demean, detrend, bandpass, form Hilbert amplitude, and smooth."""
    x = np.asarray(data, dtype=np.float64).copy()
    x -= np.mean(x)
    x = detrend(x, type="linear")
    padding_seconds = max(120.0, 5.0 / fmin, 2.0 * smooth_seconds)
    padding_samples = min(int(round(padding_seconds / dt)), len(x) - 1)
    padded = np.pad(x, (padding_samples, padding_samples), mode="reflect")
    filtered = bandpass(
        padded,
        freqmin=fmin,
        freqmax=fmax,
        df=1.0 / dt,
        corners=FILTER_CORNERS,
        zerophase=True,
    )
    filtered = filtered[padding_samples:padding_samples + len(x)]
    envelope = np.abs(hilbert(filtered))
    return gaussian_smooth(envelope, dt, smooth_seconds)


def calculate_envelope(time, data, fmin, fmax, protect_gap_edges=False):
    """Process finite segments separately; optionally mask Apollo gap edges."""
    time = np.asarray(time, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if len(time) != len(data) or len(time) < 100:
        raise ValueError("Time and data lengths differ or are too short.")

    dt = float(np.median(np.diff(time)))
    smooth_seconds = smooth_seconds_for_band(fmin, fmax)
    minimum_segment_samples = max(30, int(np.ceil(6.0 / fmin / dt)))
    envelope = np.full(data.shape, np.nan, dtype=np.float64)
    processed_count = 0

    for start, end in finite_runs(np.isfinite(data)):
        segment = data[start:end]
        if len(segment) < minimum_segment_samples:
            continue
        segment_envelope = process_segment(segment, dt, fmin, fmax, smooth_seconds)
        if protect_gap_edges:
            left_gap = start > 0
            right_gap = end < len(data)
            active_sides = int(left_gap) + int(right_gap)
            if active_sides:
                guard_seconds = max(
                    smooth_seconds, GAP_EDGE_GUARD_PERIOD_NUMBER / fmin
                )
                requested = int(round(guard_seconds / dt))
                removable = max(0, len(segment) - minimum_segment_samples)
                per_side_max = removable // active_sides
                guard = min(requested, per_side_max)
                if left_gap and guard > 0:
                    segment_envelope[:guard] = np.nan
                if right_gap and guard > 0:
                    segment_envelope[-guard:] = np.nan
        envelope[start:end] = segment_envelope
        processed_count += 1

    if processed_count == 0:
        raise ValueError(
            f"No continuous segment is long enough for {fmin:.2f}-{fmax:.2f} Hz."
        )
    return envelope


def interpolate_preserving_gaps(source_time, source_data, target_time):
    """Interpolate only within finite runs, never across a long gap."""
    source_time = np.asarray(source_time, dtype=np.float64)
    source_data = np.asarray(source_data, dtype=np.float64)
    target_time = np.asarray(target_time, dtype=np.float64)
    output = np.full(target_time.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(source_time) & np.isfinite(source_data)
    for start, end in finite_runs(valid):
        t = source_time[start:end]
        y = source_data[start:end]
        if len(t) < 2:
            continue
        inside = (target_time >= t[0]) & (target_time <= t[-1])
        if np.any(inside):
            output[inside] = np.interp(target_time[inside], t, y)
    return output


def pearson_r_zero_time(obs_time, obs_env, sim_time, sim_env):
    """Use only paired valid envelope samples in the fixed zero-shift window."""
    dt = max(
        float(np.median(np.diff(obs_time))),
        float(np.median(np.diff(sim_time))),
    )
    common_time = np.arange(
        ANALYSIS_START,
        ANALYSIS_END + 0.5 * dt,
        dt,
        dtype=np.float64,
    )
    obs = interpolate_preserving_gaps(obs_time, obs_env, common_time)
    sim = interpolate_preserving_gaps(sim_time, sim_env, common_time)
    obs_valid_count = int(np.count_nonzero(np.isfinite(obs)))
    pair_mask = np.isfinite(obs) & np.isfinite(sim)
    paired_count = int(np.count_nonzero(pair_mask))
    if obs_valid_count < 100 or paired_count < 100:
        raise ValueError("Fewer than 100 paired samples in the analysis window.")
    support_fraction = paired_count / obs_valid_count
    if support_fraction < MINIMUM_SIM_SUPPORT_FRACTION:
        raise ValueError(
            f"Simulation support {support_fraction:.3f} is below "
            f"{MINIMUM_SIM_SUPPORT_FRACTION:.2f}."
        )
    x = obs[pair_mask]
    y = sim[pair_mask]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("Invalid Pearson-r denominator.")
    r = float(np.sum(x * y) / denominator)
    return float(np.clip(r, -1.0, 1.0))


def calculate_band_r(obs_time, obs_data, sim_time, sim_data, fmin, fmax):
    """Calculate one band's observation/simulation envelope Pearson r."""
    obs_env = calculate_envelope(
        obs_time, obs_data, fmin, fmax, protect_gap_edges=True
    )
    sim_env = calculate_envelope(
        sim_time, sim_data, fmin, fmax, protect_gap_edges=False
    )
    return pearson_r_zero_time(obs_time, obs_env, sim_time, sim_env)


def build_matrices(cache_root):
    """Calculate the complete (Case, band, record) array from existing caches."""
    observation_dir = cache_root / "observations"
    simulation_root = cache_root / "simulations"
    if not observation_dir.is_dir():
        raise FileNotFoundError(f"Observation cache directory is missing: {observation_dir}")

    matrices = np.empty((len(CASE_NAMES), len(BANDS), len(RECORDS)), dtype=np.float64)
    for case_index, case_name in enumerate(CASE_NAMES):
        simulation_dir = simulation_root / case_name
        if not simulation_dir.is_dir():
            raise FileNotFoundError(
                f"Simulation cache for {case_name} is missing: {simulation_dir}"
            )
        print(f"[{case_index + 1}/{len(CASE_NAMES)}] {case_name}", flush=True)
        for record_index, (distance, event, obs_file, sim_file) in enumerate(RECORDS):
            observation_time, observation_data = load_cache(observation_dir / obs_file)
            simulation_time, simulation_data = load_cache(simulation_dir / sim_file)
            for band_index, (band_name, fmin, fmax) in enumerate(BANDS):
                try:
                    matrices[case_index, band_index, record_index] = calculate_band_r(
                        observation_time,
                        observation_data,
                        simulation_time,
                        simulation_data,
                        fmin,
                        fmax,
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"{case_name}, {distance} {event}, "
                        f"{band_name}: {error}"
                    ) from error
    return matrices


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).absolute().parent,
        help="Project directory containing data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).absolute().parent / "all_case_pearson_matrices.npz",
        help="Output NPZ path (default: beside this script)",
    )
    args = parser.parse_args(argv)
    cache_root = args.project_root / "data"
    matrices = build_matrices(cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        case_names=np.asarray(CASE_NAMES, dtype=str),
        band_names=np.asarray([item[0] for item in BANDS], dtype=str),
        distance_labels=np.asarray([item[0] for item in RECORDS], dtype=str),
        event_labels=np.asarray([item[1] for item in RECORDS], dtype=str),
        matrices=matrices,
    )
    print(f"Saved {matrices.shape} Pearson-r matrix stack: {args.output}")


if __name__ == "__main__":
    main()
