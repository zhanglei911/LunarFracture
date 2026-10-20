from pathlib import Path
from datetime import datetime
from time import perf_counter
import numpy as np
import pandas as pd
from obspy.signal.filter import bandpass
from scipy.signal import detrend, hilbert, correlate, fftconvolve

# Read the data/ directory beside this script.
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / "data"
OBS_CACHE_DIR = CACHE_ROOT / "observations"

# Select an existing simulation case directory.
CASE_NAME = "Case2"
SIM_CACHE_DIR = CACHE_ROOT / "simulations" / CASE_NAME

# Analysis outputs are written beside this public script under results/.
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / CASE_NAME
CACHE_FS = 4.0
GLOBAL_FMIN = 0.3
GLOBAL_FMAX = 0.8

# None derives Gaussian FWHM using get_smooth_seconds().
GLOBAL_SMOOTH_SECONDS = None

# All correlations use this fixed window and a strictly zero time shift.
ANALYSIS_START = 0.0
BASE_ANALYSIS_END = 4500.0
ANALYSIS_END = BASE_ANALYSIS_END
GLOBAL_MINIMUM_SIM_SUPPORT_FRACTION = 0.8
BAND_MINIMUM_SIM_SUPPORT_FRACTION = 0.65
FILTER_CORNERS = 4
MINIMUM_SMOOTH_SECONDS = 25.0
SMOOTH_PERIOD_NUMBER = 10.0
GAUSSIAN_TRUNCATE = 4.0

# Mask internal observation-gap edges; preserve a minimum number of segment samples.
ENABLE_OBSERVATION_GAP_EDGE_GUARD = True
GAP_EDGE_GUARD_PERIOD_NUMBER = 3.0

# Analysis bands in output matrix row order.
BANDS = [
    {"name": "HP3", "fmin": 0.7, "fmax": 0.8},
    {"name": "HP2", "fmin": 0.6, "fmax": 0.7},
    {"name": "HP1", "fmin": 0.5, "fmax": 0.6},
    {"name": "MP2", "fmin": 0.4, "fmax": 0.5},
    {"name": "MP1", "fmin": 0.3, "fmax": 0.4},
    {"name": "LP", "fmin": 0.2, "fmax": 0.3},
]

# Record identities, filenames, and component/channel settings used by this script.
RECORDS = [
    {
        "distance": "2.2°",
        "event_station": "A14 LM→S14",
        "obs_channel": "MHZ",
        "obs_cache_file": "A14_LM_S14_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S22_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
    {
        "distance": "2.4°",
        "event_station": "A12 LM→S12",
        "obs_channel": "MHZ",
        "obs_cache_file": "A12_LM_S12_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S24_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
    {
        "distance": "3.0°",
        "event_station": "A15 LM→S15",
        "obs_channel": "MHZ",
        "obs_cache_file": "A15_LM_S15_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S30_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
    {
        "distance": "3.8°",
        "event_station": "A14 LM→S12",
        "obs_channel": "MHZ",
        "obs_cache_file": "A14_LM_S12_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S38_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
    {
        "distance": "4.5°",
        "event_station": "A13 S-IVB→S12",
        "obs_channel": "MHZ",
        "obs_cache_file": "A13_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S45_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
    {
        "distance": "5.1°",
        "event_station": "A16 S-IVB→S12",
        "obs_channel": "MHZ",
        "obs_cache_file": "A16_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S51_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
    {
        "distance": "5.8°",
        "event_station": "A14 S-IVB→S12",
        "obs_channel": "MHZ",
        "obs_cache_file": "A14_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S58_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
    {
        "distance": "6.1°",
        "event_station": "A15 S-IVB→S14",
        "obs_channel": "MHZ",
        "obs_cache_file": "A15_SIVB_S14_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S61_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
    {
        "distance": "9.1°",
        "event_station": "A16 S-IVB→S14",
        "obs_channel": "MH1",
        "obs_cache_file": "A16_SIVB_S14_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S91_local_tangential_4Hz.npz",
        "sim_kind": "simulation_local_tangential",
    },
    {
        "distance": "11.1°",
        "event_station": "A17 S-IVB→S12",
        "obs_channel": "MHZ",
        "obs_cache_file": "A17_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S111_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
    {
        "distance": "11.7°",
        "event_station": "A15 S-IVB→S12",
        "obs_channel": "MHZ",
        "obs_cache_file": "A15_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S117_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
    {
        "distance": "25.4°",
        "event_station": "A17 LM→S15",
        "obs_channel": "MHZ",
        "obs_cache_file": "A17_LM_S15_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S254_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
    {
        "distance": "28.0°",
        "event_station": "A17 S-IVB→S16",
        "obs_channel": "MHZ",
        "obs_cache_file": "A17_SIVB_S16_minus5000_plus8000_DISP_4Hz.npz",
        "sim_cache_file": "S280_local_radial_4Hz.npz",
        "sim_kind": "simulation_local_radial",
    },
]


def elapsed_seconds(start_time):
    """Return elapsed wall-clock seconds since start_time."""
    return perf_counter() - start_time


def safe_to_csv(dataframe, output_path, index=True):
    """Write a CSV; use a timestamped filename if the destination is locked."""
    output_path = Path(output_path)
    try:
        dataframe.to_csv(output_path, index=index, encoding="utf-8-sig")
        print(f"CSV saved: {output_path}")
        return output_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alternative_path = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )
        dataframe.to_csv(alternative_path, index=index, encoding="utf-8-sig")
        print(f"Original CSV is locked; saved to: {alternative_path}")
        return alternative_path


def _npz_scalar(value):
    """Convert zero-dimensional NumPy metadata to a Python scalar."""
    value = np.asarray(value)
    if value.ndim == 0:
        return value.item()
    return value


def load_preprocessed_npz(file_path, expected_kind=None):
    """Load cache data, reconstruct its time axis, and validate sampling and kind."""
    total_start = perf_counter()
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Preprocessed cache does not exist: {file_path}")
    read_start = perf_counter()
    with np.load(file_path, allow_pickle=False) as z:
        required = {"data", "t0", "dt"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(
                f"Cache is missing required fields {sorted(missing)}: {file_path}"
            )
        data = np.asarray(z["data"], dtype=np.float64)
        t0 = float(z["t0"])
        dt = float(z["dt"])
        meta = {
            key: _npz_scalar(z[key])
            for key in z.files
            if key not in {"data", "t0", "dt"}
        }
    read_seconds = elapsed_seconds(read_start)
    if data.ndim != 1 or len(data) < 20:
        raise ValueError(
            f"Cache data must be one-dimensional with at least 20 samples: {file_path}"
        )
    if not np.isfinite(t0) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Invalid cache t0/dt: {file_path}")
    fs = 1.0 / dt
    if not np.isclose(fs, CACHE_FS, rtol=1e-07, atol=1e-10):
        raise ValueError(
            f"Unexpected cache sampling rate; expected {CACHE_FS:g} Hz; actual fs={fs:.9f} Hz; file={file_path}"
        )
    if "fs" in meta and (
        not np.isclose(float(meta["fs"]), CACHE_FS, rtol=1e-07, atol=1e-10)
    ):
        raise ValueError(
            f"Cache metadata fs={meta['fs']} differs from CACHE_FS={CACHE_FS} does not match: {file_path}"
        )
    if expected_kind is not None and "kind" in meta:
        if str(meta["kind"]) != str(expected_kind):
            raise ValueError(
                f"Cache kind mismatch: expected {expected_kind}; actual {meta['kind']}; file={file_path}"
            )
    time = t0 + np.arange(len(data), dtype=np.float64) * dt
    timing = {
        "cache_read_seconds": float(read_seconds),
        "total_cache_load_seconds": float(elapsed_seconds(total_start)),
    }
    return (time, data, meta, timing)


def _record_label(record):
    """Build the record label used by cache metadata."""
    return f"{record['distance']} {record['event_station']}"


def observation_cache_path(record):
    """Resolve the configured observation cache filename."""
    return OBS_CACHE_DIR / record["obs_cache_file"]


def simulation_cache_path(record):
    """Resolve the configured simulation cache filename."""
    return SIM_CACHE_DIR / record["sim_cache_file"]


def _validate_record_metadata(meta, record, file_path):
    """Validate record identity without repeating geometric calculations."""
    expected_label = _record_label(record)
    if "record_label" in meta:
        if str(meta["record_label"]) != expected_label:
            raise ValueError(
                f"Cache record_label mismatch: expected {expected_label}; actual {meta['record_label']}; file={file_path}"
            )
    if "distance" in meta:
        if str(meta["distance"]) != str(record["distance"]):
            raise ValueError(
                f"Cache distance mismatch: expected {record['distance']}; actual {meta['distance']}; file={file_path}"
            )


def load_cached_observation(record):
    """Load response-corrected observation displacement and check valid coverage."""
    path = observation_cache_path(record)
    time, data, meta, timing = load_preprocessed_npz(
        path, expected_kind="observation_DISP"
    )
    _validate_record_metadata(meta, record, path)
    finite = np.isfinite(data)
    valid_count = int(np.count_nonzero(finite))
    if valid_count < 100:
        raise ValueError(f"Apollo cache contains fewer than 100 valid samples: {path}")
    valid_fraction = float(np.mean(finite))
    print(
        f"    Apollo cache: {path.name}, npts={len(data):,}, fs={CACHE_FS:.3f} Hz; valid fraction={valid_fraction:.4f}; read={timing['cache_read_seconds']:.3f} s"
    )
    return (time, data, meta, timing, path)


def load_cached_simulation(record):
    """Load the configured local simulation component and validate its case."""
    path = simulation_cache_path(record)
    expected_simulation_kind = (
        "simulation_local_tangential"
        if record["distance"] == "9.1°"
        else "simulation_local_radial"
    )
    time, data, meta, timing = load_preprocessed_npz(
        path, expected_kind=expected_simulation_kind
    )
    _validate_record_metadata(meta, record, path)
    if "case_name" in meta:
        if str(meta["case_name"]) != CASE_NAME:
            raise ValueError(
                f"Simulation cache case_name={meta['case_name']}; current CASE_NAME={CASE_NAME}: {path}"
            )
    finite_count = int(np.count_nonzero(np.isfinite(data)))
    if finite_count < 100:
        raise ValueError(
            f"Simulation cache contains fewer than 100 valid samples: {path}"
        )
    print(
        f"    Simulation cache: {path.name}, npts={len(data):,}, fs={CACHE_FS:.3f} Hz; read={timing['cache_read_seconds']:.3f} s"
    )
    return (time, data, meta, timing, path)


def get_smooth_seconds(fmin, fmax):
    """Return Gaussian FWHM from the minimum width and center-frequency periods."""
    center_frequency = 0.5 * (fmin + fmax)
    return max(MINIMUM_SMOOTH_SECONDS, SMOOTH_PERIOD_NUMBER / center_frequency)


def gaussian_smooth_fft(data, dt, fwhm_seconds, truncate=4.0):
    """Smooth a nonnegative envelope by reflected-padding FFT Gaussian convolution."""
    data = np.asarray(data, dtype=np.float64)
    if len(data) < 3:
        raise ValueError("Too few samples for FFT Gaussian smoothing.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid sample interval for FFT Gaussian smoothing.")
    sigma_seconds = fwhm_seconds / 2.355
    sigma_samples = max(sigma_seconds / dt, 1.0)
    half_width = int(np.ceil(truncate * sigma_samples))
    half_width = min(half_width, len(data) - 1)
    sample_index = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (sample_index / sigma_samples) ** 2)
    kernel /= np.sum(kernel)
    padded_data = np.pad(data, (half_width, half_width), mode="reflect")
    smoothed_padded = fftconvolve(padded_data, kernel, mode="same")
    smoothed = smoothed_padded[half_width : half_width + len(data)]
    return np.maximum(smoothed, 0.0)


def _finite_runs(valid_mask):
    """Return half-open [start, end) index pairs for contiguous True runs."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.ndim != 1:
        raise ValueError("valid_mask must be a one-dimensional array.")
    padded = np.concatenate(([False], valid_mask, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return list(zip(starts, ends))


def calculate_internal_gap_edge_guard_samples(
    segment_length,
    minimum_segment_samples,
    dt,
    fmin,
    smooth_seconds,
    has_left_internal_gap,
    has_right_internal_gap,
):
    """Size gap-edge masks while retaining the minimum segment sample count."""
    segment_length = int(segment_length)
    minimum_segment_samples = int(minimum_segment_samples)
    dt = float(dt)
    fmin = float(fmin)
    smooth_seconds = float(smooth_seconds)
    if (
        not ENABLE_OBSERVATION_GAP_EDGE_GUARD
        or segment_length <= 0
        or dt <= 0.0
        or (fmin <= 0.0)
    ):
        return (0, 0, 0.0)
    left_active = bool(has_left_internal_gap)
    right_active = bool(has_right_internal_gap)
    active_side_count = int(left_active) + int(right_active)
    if active_side_count == 0:
        return (0, 0, 0.0)
    requested_seconds = max(smooth_seconds, float(GAP_EDGE_GUARD_PERIOD_NUMBER) / fmin)
    requested_samples = max(0, int(round(requested_seconds / dt)))
    removable_samples = max(0, segment_length - minimum_segment_samples)
    if active_side_count == 2:
        maximum_per_side = removable_samples // 2
    else:
        maximum_per_side = removable_samples
    applied_per_side = min(requested_samples, maximum_per_side)
    left_guard = applied_per_side if left_active else 0
    right_guard = applied_per_side if right_active else 0
    return (int(left_guard), int(right_guard), float(requested_seconds))


def _process_one_continuous_segment(segment_data, dt, fmin, fmax, smooth_seconds):
    """Demean, detrend, bandpass, form Hilbert amplitude, and Gaussian-smooth a segment."""
    segment_data = np.asarray(segment_data, dtype=np.float64)
    sampling_rate = 1.0 / dt
    stage_timing = {}
    stage_start = perf_counter()
    processed = segment_data.copy()
    processed -= np.mean(processed)
    processed = detrend(processed, type="linear")
    stage_timing["detrend_seconds"] = elapsed_seconds(stage_start)
    stage_start = perf_counter()
    padding_duration = max(120.0, 5.0 / fmin, 2.0 * smooth_seconds)
    padding_samples = min(int(round(padding_duration / dt)), len(processed) - 1)
    padded = np.pad(processed, (padding_samples, padding_samples), mode="reflect")
    stage_timing["padding_seconds"] = elapsed_seconds(stage_start)
    stage_start = perf_counter()
    filtered_padded = bandpass(
        padded,
        freqmin=fmin,
        freqmax=fmax,
        df=sampling_rate,
        corners=FILTER_CORNERS,
        zerophase=True,
    )
    filtered = filtered_padded[padding_samples : padding_samples + len(processed)]
    stage_timing["bandpass_seconds"] = elapsed_seconds(stage_start)
    stage_start = perf_counter()
    raw_envelope = np.abs(hilbert(filtered))
    stage_timing["hilbert_seconds"] = elapsed_seconds(stage_start)
    stage_start = perf_counter()
    smoothed_envelope = gaussian_smooth_fft(
        raw_envelope, dt=dt, fwhm_seconds=smooth_seconds, truncate=GAUSSIAN_TRUNCATE
    )
    stage_timing["fft_gaussian_seconds"] = elapsed_seconds(stage_start)
    return (smoothed_envelope, stage_timing)


def calculate_smoothed_envelope(
    time,
    data,
    fmin,
    fmax,
    smooth_seconds,
    timing_label="",
    apply_internal_edge_guard=False,
):
    """Process finite segments independently, optionally masking internal gap edges."""
    total_start = perf_counter()
    timing = {
        "detrend_seconds": 0.0,
        "padding_seconds": 0.0,
        "bandpass_seconds": 0.0,
        "hilbert_seconds": 0.0,
        "fft_gaussian_seconds": 0.0,
        "edge_guard_masked_samples": 0.0,
        "edge_guard_segment_count": 0.0,
        "edge_guard_requested_seconds": 0.0,
    }
    time = np.asarray(time, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if time.ndim != 1 or data.ndim != 1 or len(time) != len(data):
        raise ValueError(
            "time and data must be one-dimensional arrays of equal length."
        )
    if len(time) < 20:
        raise ValueError("Envelope calculation requires at least 20 samples.")
    dt = float(np.median(np.diff(time)))
    sampling_rate = 1.0 / dt
    nyquist = 0.5 * sampling_rate
    if fmax >= 0.95 * nyquist:
        raise ValueError(
            f"Band upper limit {fmax:.2f} Hz approaches or exceeds Nyquist={nyquist:.2f} Hz."
        )
    prefix = f"{timing_label}: " if timing_label else ""
    envelope = np.full(data.shape, np.nan, dtype=np.float64)
    valid_mask = np.isfinite(data)
    runs = _finite_runs(valid_mask)
    minimum_segment_samples = max(30, int(np.ceil(6.0 / fmin / dt)))
    processed_segment_count = 0
    skipped_segment_count = 0
    for start_index, end_index in runs:
        segment = data[start_index:end_index]
        if len(segment) < minimum_segment_samples:
            skipped_segment_count += 1
            continue
        segment_envelope, segment_timing = _process_one_continuous_segment(
            segment_data=segment,
            dt=dt,
            fmin=fmin,
            fmax=fmax,
            smooth_seconds=smooth_seconds,
        )
        for stage_name, stage_seconds in segment_timing.items():
            timing[stage_name] += stage_seconds
        if apply_internal_edge_guard:
            left_guard, right_guard, requested_guard_seconds = (
                calculate_internal_gap_edge_guard_samples(
                    segment_length=len(segment),
                    minimum_segment_samples=minimum_segment_samples,
                    dt=dt,
                    fmin=fmin,
                    smooth_seconds=smooth_seconds,
                    has_left_internal_gap=start_index > 0,
                    has_right_internal_gap=end_index < len(data),
                )
            )
            if left_guard > 0:
                segment_envelope[:left_guard] = np.nan
            if right_guard > 0:
                segment_envelope[-right_guard:] = np.nan
            masked_here = int(left_guard + right_guard)
            timing["edge_guard_masked_samples"] += masked_here
            timing["edge_guard_requested_seconds"] = max(
                timing["edge_guard_requested_seconds"], requested_guard_seconds
            )
            if masked_here > 0:
                timing["edge_guard_segment_count"] += 1.0
        envelope[start_index:end_index] = segment_envelope
        processed_segment_count += 1
    if processed_segment_count == 0:
        raise ValueError(
            f"{timing_label}No continuous segments are long enough for {fmin:.2f}-{fmax:.2f} Hz envelope calculation."
        )
    timing["valid_envelope_fraction"] = float(np.mean(np.isfinite(envelope)))
    timing["processed_segment_count"] = float(processed_segment_count)
    timing["skipped_short_segment_count"] = float(skipped_segment_count)
    timing["total_envelope_seconds"] = elapsed_seconds(total_start)
    print(
        f"        {prefix}npts={len(data):,}, fs={sampling_rate:.3f} Hz; continuous segments={processed_segment_count}; edge guard={int(timing['edge_guard_masked_samples']):,} samples; valid envelope fraction={timing['valid_envelope_fraction']:.4f}; total={timing['total_envelope_seconds']:.3f} s"
    )
    return (envelope, timing)


def _interpolate_preserving_gaps(source_time, source_data, target_time):
    """Interpolate finite runs separately so internal gaps remain NaN."""
    source_time = np.asarray(source_time, dtype=np.float64)
    source_data = np.asarray(source_data, dtype=np.float64)
    target_time = np.asarray(target_time, dtype=np.float64)
    output = np.full(target_time.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(source_time) & np.isfinite(source_data)
    for start_index, end_index in _finite_runs(valid):
        segment_time = source_time[start_index:end_index]
        segment_data = source_data[start_index:end_index]
        if len(segment_time) < 2:
            continue
        inside = (target_time >= segment_time[0]) & (target_time <= segment_time[-1])
        if np.any(inside):
            output[inside] = np.interp(target_time[inside], segment_time, segment_data)
    return output


def normalize_nonnegative_envelope(envelope, name):
    """Normalize finite nonnegative envelope values by their maximum."""
    envelope = np.asarray(envelope, dtype=np.float64)
    if envelope.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    valid = np.isfinite(envelope)
    if np.count_nonzero(valid) < 2:
        raise ValueError(f"{name} has insufficient valid samples.")
    normalized = np.full(envelope.shape, np.nan, dtype=np.float64)
    finite_values = np.maximum(envelope[valid], 0.0)
    maximum = float(np.max(finite_values))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"{name} has zero maximum and cannot be normalized.")
    normalized[valid] = finite_values / maximum
    return normalized


def make_fixed_time_axis(start_time, end_time, dt):
    """Construct the fixed analysis grid using the supplied sample interval."""
    start_time = float(start_time)
    end_time = float(end_time)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Invalid common sample interval.")
    if end_time <= start_time:
        raise ValueError(f"Invalid time window: {start_time:.3f}–{end_time:.3f} s.")
    number_of_samples = int(np.floor((end_time - start_time) / dt)) + 1
    if number_of_samples < 100:
        raise ValueError("The fixed analysis window contains fewer than 100 samples.")
    return start_time + np.arange(number_of_samples) * dt


def search_best_lag_in_fixed_window(
    observation_time,
    observation_envelope,
    simulation_time,
    simulation_envelope,
    window_start,
    window_end,
    center_lag_seconds,
    minimum_residual_lag_seconds,
    maximum_residual_lag_seconds,
    minimum_support_fraction,
):
    """Evaluate masked Pearson r over the supplied lag range.

    The analysis callers pass zero for both lag limits and the center lag,
    so only zero shift is evaluated. Preserve the original FFT sums and
    simulation-support convention for numerical compatibility.
    """
    observation_time = np.asarray(observation_time, dtype=np.float64)
    observation_envelope = np.asarray(observation_envelope, dtype=np.float64)
    simulation_time = np.asarray(simulation_time, dtype=np.float64)
    simulation_envelope = np.asarray(simulation_envelope, dtype=np.float64)
    observation_dt = float(np.median(np.diff(observation_time)))
    simulation_dt = float(np.median(np.diff(simulation_time)))
    common_dt = max(observation_dt, simulation_dt)
    fixed_time = make_fixed_time_axis(window_start, window_end, common_dt)
    observation_window = _interpolate_preserving_gaps(
        source_time=observation_time,
        source_data=observation_envelope,
        target_time=fixed_time,
    )
    observation_window = normalize_nonnegative_envelope(
        observation_window, "Observation envelope in the fixed time window"
    )
    observation_mask = np.isfinite(observation_window).astype(np.float64)
    observation_filled = np.where(
        np.isfinite(observation_window), observation_window, 0.0
    )
    observation_squared_filled = observation_filled**2
    valid_observation_samples = int(np.sum(observation_mask))
    if valid_observation_samples < 100:
        raise ValueError(
            "Fewer than 100 valid observation samples in the fixed analysis window."
        )
    minimum_residual_lag_seconds = float(minimum_residual_lag_seconds)
    maximum_residual_lag_seconds = float(maximum_residual_lag_seconds)
    if not np.isfinite(minimum_residual_lag_seconds) or not np.isfinite(
        maximum_residual_lag_seconds
    ):
        raise ValueError("Residual lag bounds must be finite.")
    if minimum_residual_lag_seconds > maximum_residual_lag_seconds:
        raise ValueError(
            f"Residual lag lower bound exceeds the upper bound: {minimum_residual_lag_seconds} > {maximum_residual_lag_seconds}"
        )
    maximum_extension_seconds = max(
        abs(minimum_residual_lag_seconds), abs(maximum_residual_lag_seconds)
    )
    maximum_extension_samples = int(np.ceil(maximum_extension_seconds / common_dt))
    number_of_samples = len(fixed_time)
    centered_simulation_time = simulation_time + float(center_lag_seconds)
    extended_index = np.arange(
        number_of_samples + 2 * maximum_extension_samples, dtype=np.float64
    )
    extended_time = (
        float(window_start) + (extended_index - maximum_extension_samples) * common_dt
    )
    simulation_extended = np.interp(
        extended_time,
        centered_simulation_time,
        np.nan_to_num(simulation_envelope, nan=0.0),
        left=0.0,
        right=0.0,
    )
    simulation_extended = np.maximum(simulation_extended, 0.0)
    simulation_support_extended = (
        (extended_time >= centered_simulation_time[0])
        & (extended_time <= centered_simulation_time[-1])
    ).astype(np.float64)
    overlap_samples = correlate(
        simulation_support_extended, observation_mask, mode="valid", method="fft"
    )
    overlap_samples = np.rint(np.maximum(overlap_samples, 0.0))
    sum_xy = correlate(
        simulation_extended, observation_filled, mode="valid", method="fft"
    )
    sum_x = correlate(
        simulation_support_extended, observation_filled, mode="valid", method="fft"
    )
    sum_x2 = correlate(
        simulation_support_extended,
        observation_squared_filled,
        mode="valid",
        method="fft",
    )
    sum_y = correlate(simulation_extended, observation_mask, mode="valid", method="fft")
    sum_y2 = correlate(
        simulation_extended**2, observation_mask, mode="valid", method="fft"
    )
    pearson_r = np.full_like(sum_xy, np.nan, dtype=np.float64)
    enough_pairs = overlap_samples >= 3.0
    safe_n = np.where(enough_pairs, overlap_samples, np.nan)
    pearson_numerator = sum_xy - sum_x * sum_y / safe_n
    centered_x_energy = sum_x2 - sum_x**2 / safe_n
    centered_y_energy = sum_y2 - sum_y**2 / safe_n
    centered_x_energy = np.maximum(centered_x_energy, 0.0)
    centered_y_energy = np.maximum(centered_y_energy, 0.0)
    pearson_denominator = np.sqrt(centered_x_energy * centered_y_energy)
    valid_denominator = (
        enough_pairs & np.isfinite(pearson_denominator) & (pearson_denominator > 0.0)
    )
    pearson_r[valid_denominator] = (
        pearson_numerator[valid_denominator] / pearson_denominator[valid_denominator]
    )
    pearson_r[valid_denominator] = np.clip(pearson_r[valid_denominator], -1.0, 1.0)
    segment_start_indices = np.arange(len(pearson_r))
    residual_lag_samples = maximum_extension_samples - segment_start_indices
    residual_lag_seconds = residual_lag_samples * common_dt
    absolute_lag_seconds = float(center_lag_seconds) + residual_lag_seconds
    support_fraction = overlap_samples / valid_observation_samples
    valid = np.isfinite(pearson_r)
    valid &= support_fraction >= float(minimum_support_fraction)
    lag_tolerance = 0.5 * common_dt + 1e-12
    valid &= residual_lag_seconds >= minimum_residual_lag_seconds - lag_tolerance
    valid &= residual_lag_seconds <= maximum_residual_lag_seconds + lag_tolerance
    candidate = pearson_r.copy()
    candidate[~valid] = np.nan
    if np.all(~np.isfinite(candidate)):
        raise ValueError(
            "No valid Pearson r meets the lag bounds, observation mask, and minimum simulation coverage."
        )
    best_index = int(np.nanargmax(candidate))
    best_absolute_lag = float(absolute_lag_seconds[best_index])
    best_shifted_simulation_time = simulation_time + best_absolute_lag
    best_simulation_window = np.interp(
        fixed_time,
        best_shifted_simulation_time,
        np.nan_to_num(simulation_envelope, nan=0.0),
        left=np.nan,
        right=np.nan,
    )
    finite_best_sim = np.isfinite(best_simulation_window)
    if np.count_nonzero(finite_best_sim) >= 2:
        finite_values = np.maximum(best_simulation_window[finite_best_sim], 0.0)
        max_value = float(np.max(finite_values))
        if np.isfinite(max_value) and max_value > 0.0:
            best_simulation_window[finite_best_sim] = finite_values / max_value
    return {
        "max_pearson_r": float(candidate[best_index]),
        "absolute_lag_seconds": best_absolute_lag,
        "residual_lag_seconds": float(residual_lag_seconds[best_index]),
        "center_lag_seconds": float(center_lag_seconds),
        "support_fraction": float(support_fraction[best_index]),
        "paired_samples": int(overlap_samples[best_index]),
        "common_dt": float(common_dt),
        "window_start": float(fixed_time[0]),
        "window_end": float(fixed_time[-1]),
        "number_of_samples": int(number_of_samples),
        "valid_observation_samples": valid_observation_samples,
        "observation_valid_fraction": float(
            valid_observation_samples / number_of_samples
        ),
        "fixed_time": fixed_time,
        "observation_window": observation_window,
        "simulation_window": best_simulation_window,
    }


def calculate_global_alignment(
    simulation_time, simulation_data, observation_time, observation_data
):
    """Evaluate global-band Pearson r at zero shift in the configured window."""
    total_start = perf_counter()
    if GLOBAL_SMOOTH_SECONDS is None:
        smooth_seconds = get_smooth_seconds(GLOBAL_FMIN, GLOBAL_FMAX)
    else:
        smooth_seconds = float(GLOBAL_SMOOTH_SECONDS)
    simulation_envelope_full, simulation_timing = calculate_smoothed_envelope(
        simulation_time,
        simulation_data,
        GLOBAL_FMIN,
        GLOBAL_FMAX,
        smooth_seconds,
        timing_label="Simulation/global band/zero shift",
    )
    observation_envelope, observation_timing = calculate_smoothed_envelope(
        observation_time,
        observation_data,
        GLOBAL_FMIN,
        GLOBAL_FMAX,
        smooth_seconds,
        timing_label="Observation/global band/zero shift",
        apply_internal_edge_guard=True,
    )
    analysis_start = float(ANALYSIS_START)
    analysis_end = float(ANALYSIS_END)
    observation_dt = float(np.median(np.diff(observation_time)))
    simulation_dt = float(np.median(np.diff(simulation_time)))
    if observation_time[0] - 0.5 * observation_dt > analysis_start:
        raise ValueError(
            f"Apollo record starts at {observation_time[0]:.3f} s and cannot cover the fixed analysis start at {analysis_start:.1f} s."
        )
    if observation_time[-1] + 0.5 * observation_dt < analysis_end:
        raise ValueError(
            f"Apollo record ends at {observation_time[-1]:.3f} s and cannot cover the fixed analysis end at {analysis_end:.1f} s."
        )
    if simulation_time[0] - 0.5 * simulation_dt > analysis_start:
        raise ValueError(
            f"Simulation record starts at {simulation_time[0]:.3f} s and cannot cover the fixed analysis start at {analysis_start:.1f} s."
        )
    if simulation_time[-1] + 0.5 * simulation_dt < analysis_end:
        raise ValueError(
            f"Simulation record ends at {simulation_time[-1]:.3f} s and cannot cover the fixed analysis end at {analysis_end:.1f} s."
        )
    evaluation_start = perf_counter()
    search_result = search_best_lag_in_fixed_window(
        observation_time=observation_time,
        observation_envelope=observation_envelope,
        simulation_time=simulation_time,
        simulation_envelope=simulation_envelope_full,
        window_start=analysis_start,
        window_end=analysis_end,
        center_lag_seconds=0.0,
        minimum_residual_lag_seconds=0.0,
        maximum_residual_lag_seconds=0.0,
        minimum_support_fraction=GLOBAL_MINIMUM_SIM_SUPPORT_FRACTION,
    )
    evaluation_seconds = elapsed_seconds(evaluation_start)
    returned_shift = float(search_result["absolute_lag_seconds"])
    tolerance = 0.5 * float(search_result["common_dt"]) + 1e-10
    if abs(returned_shift) > tolerance:
        raise RuntimeError(
            f"Unexpected zero-shift result: shift={returned_shift:+.6f} s."
        )
    support = float(search_result["support_fraction"])
    if support < 0.85:
        print(
            f"    Note: Paired valid coverage within the configured analysis window is {support:.3f}."
        )
    total_seconds = elapsed_seconds(total_start)
    print(
        f"    GLOBAL band={GLOBAL_FMIN:.2f}–{GLOBAL_FMAX:.2f} Hz, smooth={smooth_seconds:.1f} s"
    )
    print(
        f"    Zero-shift Pearson r={search_result['max_pearson_r']:.4f}, time shift=+0.0 s"
    )
    print(f"    Fixed analysis window={analysis_start:.1f}–{analysis_end:.1f} s")
    return {
        "alignment_mode": "ZERO_TIME",
        "global_fmin": GLOBAL_FMIN,
        "global_fmax": GLOBAL_FMAX,
        "global_smooth_seconds": smooth_seconds,
        "global_pearson_r": search_result["max_pearson_r"],
        "global_lag_seconds": 0.0,
        "applied_global_lag_seconds": 0.0,
        "lag_source": "ZERO_TIME",
        "manual_global_lag_seconds": np.nan,
        "auto_global_lag_seconds": np.nan,
        "auto_global_pearson_r": np.nan,
        "global_support_fraction": search_result["support_fraction"],
        "global_paired_samples": search_result["paired_samples"],
        "global_search_start": analysis_start,
        "global_search_end": analysis_end,
        "global_search_number_of_samples": search_result["number_of_samples"],
        "global_valid_observation_samples": search_result["valid_observation_samples"],
        "global_observation_valid_fraction": search_result[
            "observation_valid_fraction"
        ],
        "base_analysis_end": float(BASE_ANALYSIS_END),
        "analysis_reference_shifted_start": 0.0,
        "analysis_reference_shifted_end": analysis_end,
        "shifted_simulation_start": float(simulation_time[0]),
        "shifted_simulation_end": float(simulation_time[-1]),
        "full_shifted_simulation_start": float(simulation_time[0]),
        "full_shifted_simulation_end": float(simulation_time[-1]),
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
        "analysis_duration_seconds": analysis_end - analysis_start,
        "simulation_envelope_timing": simulation_timing,
        "observation_envelope_timing": observation_timing,
        "lag_search_seconds": evaluation_seconds,
        "total_global_seconds": total_seconds,
    }


def calculate_band_max_pearson_r(
    simulation_time,
    simulation_data,
    observation_time,
    observation_data,
    fmin,
    fmax,
    global_lag_seconds=0.0,
    analysis_start=ANALYSIS_START,
    analysis_end=ANALYSIS_END,
):
    """Evaluate bandwise Pearson r at zero shift; reject nonzero global lag."""
    total_start = perf_counter()
    if abs(float(global_lag_seconds)) > 1e-12:
        raise ValueError("Zero-time alignment requires global_lag_seconds=0.")
    analysis_start = float(analysis_start)
    analysis_end = float(analysis_end)
    if not np.isclose(analysis_start, ANALYSIS_START, atol=1e-12):
        raise ValueError(f"The fixed analysis start must be {ANALYSIS_START:.1f} s.")
    if not np.isclose(analysis_end, ANALYSIS_END, atol=1e-12):
        raise ValueError(f"The fixed analysis end must be {ANALYSIS_END:.1f} s.")
    smooth_seconds = get_smooth_seconds(fmin, fmax)
    simulation_envelope_full, simulation_timing = calculate_smoothed_envelope(
        simulation_time,
        simulation_data,
        fmin,
        fmax,
        smooth_seconds,
        timing_label="Simulation/zero shift",
    )
    observation_envelope, observation_timing = calculate_smoothed_envelope(
        observation_time,
        observation_data,
        fmin,
        fmax,
        smooth_seconds,
        timing_label="Observation/zero shift",
        apply_internal_edge_guard=True,
    )
    evaluation_start = perf_counter()
    search_result = search_best_lag_in_fixed_window(
        observation_time=observation_time,
        observation_envelope=observation_envelope,
        simulation_time=simulation_time,
        simulation_envelope=simulation_envelope_full,
        window_start=analysis_start,
        window_end=analysis_end,
        center_lag_seconds=0.0,
        minimum_residual_lag_seconds=0.0,
        maximum_residual_lag_seconds=0.0,
        minimum_support_fraction=BAND_MINIMUM_SIM_SUPPORT_FRACTION,
    )
    evaluation_seconds = elapsed_seconds(evaluation_start)
    total_band_seconds = elapsed_seconds(total_start)
    returned_shift = float(search_result["absolute_lag_seconds"])
    tolerance = 0.5 * float(search_result["common_dt"]) + 1e-10
    if abs(returned_shift) > tolerance:
        raise RuntimeError(
            f"Unexpected bandwise zero-shift result: shift={returned_shift:+.6f} s."
        )
    print(
        f"        Pearson r@zero time={search_result['max_pearson_r']:.4f}, window={analysis_start:.1f}–{analysis_end:.1f} s"
    )
    return {
        "alignment_mode": "ZERO_TIME",
        "smooth_seconds": smooth_seconds,
        "max_pearson_r": search_result["max_pearson_r"],
        "best_lag_seconds": 0.0,
        "delta_lag_seconds": 0.0,
        "global_lag_seconds": 0.0,
        "base_analysis_end": float(BASE_ANALYSIS_END),
        "support_fraction": search_result["support_fraction"],
        "paired_samples": search_result["paired_samples"],
        "common_start": search_result["window_start"],
        "common_end": search_result["window_end"],
        "common_dt": search_result["common_dt"],
        "number_of_samples": search_result["number_of_samples"],
        "valid_observation_samples": search_result["valid_observation_samples"],
        "observation_valid_fraction": search_result["observation_valid_fraction"],
        "simulation_envelope_timing": simulation_timing,
        "observation_envelope_timing": observation_timing,
        "lag_search_seconds": evaluation_seconds,
        "total_band_seconds": total_band_seconds,
    }


def main():
    """Run the configured processing and write numerical outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    program_start = perf_counter()
    if not OBS_CACHE_DIR.exists():
        raise FileNotFoundError(
            f"Apollo cache directory does not exist: {OBS_CACHE_DIR}\nRun preprocess_apollo_observations_original_segmented_gap_13records_4Hz.py first."
        )
    if not SIM_CACHE_DIR.exists():
        raise FileNotFoundError(
            f"Simulation cache directory does not exist: {SIM_CACHE_DIR}\nGenerate the simulation caches for CASE_NAME or update CASE_NAME."
        )
    print("=" * 88)
    print(f"Cached-data Pearson correlation: Case={CASE_NAME}")
    print("=" * 88)
    print(f"Apollo cache directory: {OBS_CACHE_DIR}")
    print(f"Simulation cache directory: {SIM_CACHE_DIR}")
    print(f"Record count: {len(RECORDS)}")
    print(f"GLOBAL band: {GLOBAL_FMIN:.2f}–{GLOBAL_FMAX:.2f} Hz")
    print(
        "Alignment: Apollo impact time t=0 corresponds directly to SPECFEM2D source time t=0"
    )
    print(
        f"Fixed analysis window for all bands: {ANALYSIS_START:.1f}–{ANALYSIS_END:.1f} s; time shift is strictly 0.0 s"
    )
    number_of_bands = len(BANDS)
    number_of_records = len(RECORDS)
    maximum_pearson_r_matrix = np.full((number_of_bands, number_of_records), np.nan)
    best_lag_matrix = np.full_like(maximum_pearson_r_matrix, np.nan)
    delta_lag_matrix = np.full_like(maximum_pearson_r_matrix, np.nan)
    global_lag_vector = np.full(number_of_records, np.nan)
    global_pearson_r_vector = np.full(number_of_records, np.nan)
    analysis_end_vector = np.full(number_of_records, np.nan)
    detailed_results = []
    global_results = []
    timing_log = []
    for record_index, record in enumerate(RECORDS):
        record_label = _record_label(record)
        observation_path = observation_cache_path(record)
        simulation_path = simulation_cache_path(record)
        print("\n" + "=" * 88)
        print(f"[{record_index + 1}/{number_of_records}] {record_label}")
        print(f"Apollo cache: {observation_path}")
        print(f"Simulation cache: {simulation_path}")
        try:
            (
                observation_time,
                observation_data,
                observation_meta,
                observation_cache_timing,
                observation_cache_file,
            ) = load_cached_observation(record)
            (
                simulation_time,
                simulation_data,
                simulation_meta,
                simulation_cache_timing,
                simulation_cache_file,
            ) = load_cached_simulation(record)
            for stage, seconds in observation_cache_timing.items():
                timing_log.append(
                    {
                        "record": record_label,
                        "band": "ALL",
                        "data_type": "observation_cache",
                        "stage": stage,
                        "seconds": seconds,
                    }
                )
            for stage, seconds in simulation_cache_timing.items():
                timing_log.append(
                    {
                        "record": record_label,
                        "band": "ALL",
                        "data_type": "simulation_cache",
                        "stage": stage,
                        "seconds": seconds,
                    }
                )
            print(
                "\n    Computing the global-band Pearson coefficient at zero time shift."
            )
            auto_global_result = calculate_global_alignment(
                simulation_time=simulation_time,
                simulation_data=simulation_data,
                observation_time=observation_time,
                observation_data=observation_data,
            )
            global_result = auto_global_result
            print(
                f"    Alignment=ZERO_TIME, time shift=+0.0 s, analysis window={global_result['analysis_start']:.1f}–{global_result['analysis_end']:.1f} s"
            )
            global_lag_vector[record_index] = global_result["global_lag_seconds"]
            global_pearson_r_vector[record_index] = global_result["global_pearson_r"]
            analysis_end_vector[record_index] = global_result["analysis_end"]
            global_results.append(
                {
                    "distance": record["distance"],
                    "event_station": record["event_station"],
                    "case_name": CASE_NAME,
                    "alignment_mode": "ZERO_TIME",
                    "observation_cache_file": str(observation_cache_file),
                    "simulation_cache_file": str(simulation_cache_file),
                    "cache_fs_hz": CACHE_FS,
                    "global_fmin": global_result["global_fmin"],
                    "global_fmax": global_result["global_fmax"],
                    "global_smooth_seconds": global_result["global_smooth_seconds"],
                    "global_pearson_r": global_result["global_pearson_r"],
                    "global_lag_seconds": global_result["global_lag_seconds"],
                    "applied_global_lag_seconds": global_result[
                        "applied_global_lag_seconds"
                    ],
                    "lag_source": global_result["lag_source"],
                    "manual_global_lag_seconds": global_result[
                        "manual_global_lag_seconds"
                    ],
                    "auto_global_lag_seconds": global_result["auto_global_lag_seconds"],
                    "auto_global_pearson_r": global_result["auto_global_pearson_r"],
                    "global_support_fraction": global_result["global_support_fraction"],
                    "global_paired_samples": global_result["global_paired_samples"],
                    "global_search_start": global_result["global_search_start"],
                    "global_search_end": global_result["global_search_end"],
                    "global_valid_observation_samples": global_result[
                        "global_valid_observation_samples"
                    ],
                    "global_observation_valid_fraction": global_result[
                        "global_observation_valid_fraction"
                    ],
                    "base_analysis_end": global_result["base_analysis_end"],
                    "full_shifted_simulation_start": global_result[
                        "full_shifted_simulation_start"
                    ],
                    "full_shifted_simulation_end": global_result[
                        "full_shifted_simulation_end"
                    ],
                    "analysis_start": global_result["analysis_start"],
                    "analysis_end": global_result["analysis_end"],
                    "analysis_duration_seconds": global_result[
                        "analysis_duration_seconds"
                    ],
                    "status": "ok",
                }
            )
            global_sim_timing = global_result["simulation_envelope_timing"]
            global_obs_timing = global_result["observation_envelope_timing"]
            for stage, seconds in global_sim_timing.items():
                timing_log.append(
                    {
                        "record": record_label,
                        "band": f"GLOBAL {GLOBAL_FMIN:.2f}-{GLOBAL_FMAX:.2f} Hz",
                        "data_type": "simulation_global_envelope",
                        "stage": stage,
                        "seconds": seconds,
                    }
                )
            for stage, seconds in global_obs_timing.items():
                timing_log.append(
                    {
                        "record": record_label,
                        "band": f"GLOBAL {GLOBAL_FMIN:.2f}-{GLOBAL_FMAX:.2f} Hz",
                        "data_type": "observation_global_envelope",
                        "stage": stage,
                        "seconds": seconds,
                    }
                )
            timing_log.extend(
                [
                    {
                        "record": record_label,
                        "band": f"GLOBAL {GLOBAL_FMIN:.2f}-{GLOBAL_FMAX:.2f} Hz",
                        "data_type": "comparison",
                        "stage": "global_zero_time_pearson",
                        "seconds": global_result["lag_search_seconds"],
                    },
                    {
                        "record": record_label,
                        "band": f"GLOBAL {GLOBAL_FMIN:.2f}-{GLOBAL_FMAX:.2f} Hz",
                        "data_type": "comparison",
                        "stage": "global_total",
                        "seconds": global_result["total_global_seconds"],
                    },
                ]
            )
        except Exception as error:
            print(f"    Cache loading or global calculation failed: {error}")
            global_results.append(
                {
                    "distance": record["distance"],
                    "event_station": record["event_station"],
                    "case_name": CASE_NAME,
                    "observation_cache_file": str(observation_path),
                    "simulation_cache_file": str(simulation_path),
                    "status": f"global_failed: {error}",
                }
            )
            for band in BANDS:
                detailed_results.append(
                    {
                        "distance": record["distance"],
                        "event_station": record["event_station"],
                        "case_name": CASE_NAME,
                        "observation_cache_file": str(observation_path),
                        "simulation_cache_file": str(simulation_path),
                        "band_name": band["name"],
                        "fmin": band["fmin"],
                        "fmax": band["fmax"],
                        "status": f"cache_load_or_global_failed: {error}",
                    }
                )
            continue
        for band_index, band in enumerate(BANDS):
            fmin = band["fmin"]
            fmax = band["fmax"]
            band_label = f"{band['name']} {fmin:.2f}-{fmax:.2f} Hz"
            print(f"\n    [{band_index + 1}/{number_of_bands}] {band_label}")
            try:
                result = calculate_band_max_pearson_r(
                    simulation_time=simulation_time,
                    simulation_data=simulation_data,
                    observation_time=observation_time,
                    observation_data=observation_data,
                    fmin=fmin,
                    fmax=fmax,
                    global_lag_seconds=global_result["global_lag_seconds"],
                    analysis_start=global_result["analysis_start"],
                    analysis_end=global_result["analysis_end"],
                )
                maximum_pearson_r_matrix[band_index, record_index] = result[
                    "max_pearson_r"
                ]
                best_lag_matrix[band_index, record_index] = result["best_lag_seconds"]
                delta_lag_matrix[band_index, record_index] = result["delta_lag_seconds"]
                sim_env_timing = result["simulation_envelope_timing"]
                obs_env_timing = result["observation_envelope_timing"]
                detailed_results.append(
                    {
                        "distance": record["distance"],
                        "event_station": record["event_station"],
                        "case_name": CASE_NAME,
                        "observation_cache_file": str(observation_cache_file),
                        "simulation_cache_file": str(simulation_cache_file),
                        "cache_fs_hz": CACHE_FS,
                        "band_name": band["name"],
                        "fmin": fmin,
                        "fmax": fmax,
                        "smooth_seconds": result["smooth_seconds"],
                        "max_pearson_r": result["max_pearson_r"],
                        "global_pearson_r": global_result["global_pearson_r"],
                        "lag_source": global_result["lag_source"],
                        "manual_global_lag_seconds": global_result[
                            "manual_global_lag_seconds"
                        ],
                        "auto_global_lag_seconds": global_result[
                            "auto_global_lag_seconds"
                        ],
                        "auto_global_pearson_r": global_result["auto_global_pearson_r"],
                        "global_lag_seconds": result["global_lag_seconds"],
                        "best_lag_seconds": result["best_lag_seconds"],
                        "delta_lag_seconds": result["delta_lag_seconds"],
                        "support_fraction": result["support_fraction"],
                        "paired_samples": result["paired_samples"],
                        "common_start": result["common_start"],
                        "common_end": result["common_end"],
                        "common_dt": result["common_dt"],
                        "number_of_samples": result["number_of_samples"],
                        "valid_observation_samples": result[
                            "valid_observation_samples"
                        ],
                        "observation_valid_fraction": result[
                            "observation_valid_fraction"
                        ],
                        "base_analysis_end": global_result["base_analysis_end"],
                        "full_shifted_simulation_end": global_result[
                            "full_shifted_simulation_end"
                        ],
                        "sim_bandpass_seconds": sim_env_timing["bandpass_seconds"],
                        "sim_hilbert_seconds": sim_env_timing["hilbert_seconds"],
                        "sim_fft_gaussian_seconds": sim_env_timing[
                            "fft_gaussian_seconds"
                        ],
                        "sim_total_envelope_seconds": sim_env_timing[
                            "total_envelope_seconds"
                        ],
                        "obs_bandpass_seconds": obs_env_timing["bandpass_seconds"],
                        "obs_hilbert_seconds": obs_env_timing["hilbert_seconds"],
                        "obs_fft_gaussian_seconds": obs_env_timing[
                            "fft_gaussian_seconds"
                        ],
                        "obs_total_envelope_seconds": obs_env_timing[
                            "total_envelope_seconds"
                        ],
                        "lag_search_seconds": result["lag_search_seconds"],
                        "total_band_seconds": result["total_band_seconds"],
                        "simulation_component": (
                            "local_tangential_4Hz_cache"
                            if record["distance"] == "9.1°"
                            else "local_radial_4Hz_cache"
                        ),
                        "observation_component": "DISP_4Hz_cache",
                        "status": "ok",
                    }
                )
                for stage, seconds in sim_env_timing.items():
                    timing_log.append(
                        {
                            "record": record_label,
                            "band": band_label,
                            "data_type": "simulation_envelope",
                            "stage": stage,
                            "seconds": seconds,
                        }
                    )
                for stage, seconds in obs_env_timing.items():
                    timing_log.append(
                        {
                            "record": record_label,
                            "band": band_label,
                            "data_type": "observation_envelope",
                            "stage": stage,
                            "seconds": seconds,
                        }
                    )
                timing_log.extend(
                    [
                        {
                            "record": record_label,
                            "band": band_label,
                            "data_type": "comparison",
                            "stage": "band_zero_time_pearson",
                            "seconds": result["lag_search_seconds"],
                        },
                        {
                            "record": record_label,
                            "band": band_label,
                            "data_type": "comparison",
                            "stage": "total_band",
                            "seconds": result["total_band_seconds"],
                        },
                    ]
                )
            except Exception as error:
                print(f"        Band calculation failed: {error}")
                detailed_results.append(
                    {
                        "distance": record["distance"],
                        "event_station": record["event_station"],
                        "case_name": CASE_NAME,
                        "observation_cache_file": str(observation_cache_file),
                        "simulation_cache_file": str(simulation_cache_file),
                        "band_name": band["name"],
                        "fmin": fmin,
                        "fmax": fmax,
                        "global_lag_seconds": global_result["global_lag_seconds"],
                        "status": f"calculation_failed: {error}",
                    }
                )
    detailed_dataframe = pd.DataFrame(detailed_results)
    global_dataframe = pd.DataFrame(global_results)
    detailed_csv = safe_to_csv(
        detailed_dataframe,
        OUTPUT_DIR / "zero_time_band_pearson_detailed_results.csv",
        index=False,
    )
    global_csv = safe_to_csv(
        global_dataframe, OUTPUT_DIR / "zero_time_global_band_results.csv", index=False
    )
    row_labels = [
        f"{band['name']} {band['fmin']:.2f}-{band['fmax']:.2f}" for band in BANDS
    ]
    column_labels = [
        f"{record['distance']} {record['event_station']}" for record in RECORDS
    ]
    max_pearson_r_dataframe = pd.DataFrame(
        maximum_pearson_r_matrix, index=row_labels, columns=column_labels
    )
    max_pearson_r_csv = safe_to_csv(
        max_pearson_r_dataframe,
        OUTPUT_DIR / "zero_time_band_pearson_r_matrix.csv",
        index=True,
    )
    best_lag_dataframe = pd.DataFrame(
        best_lag_matrix, index=row_labels, columns=column_labels
    )
    best_lag_csv = safe_to_csv(
        best_lag_dataframe,
        OUTPUT_DIR / "zero_time_shift_seconds_matrix_should_be_zero.csv",
        index=True,
    )
    delta_lag_dataframe = pd.DataFrame(
        delta_lag_matrix, index=row_labels, columns=column_labels
    )
    delta_lag_csv = safe_to_csv(
        delta_lag_dataframe,
        OUTPUT_DIR / "zero_time_delta_shift_seconds_matrix_should_be_zero.csv",
        index=True,
    )
    global_summary_dataframe = pd.DataFrame(
        {
            "base_analysis_end_seconds": np.full(number_of_records, BASE_ANALYSIS_END),
            "alignment_mode": np.full(number_of_records, "ZERO_TIME", dtype=object),
            "time_shift_seconds": np.zeros(number_of_records, dtype=np.float64),
            "global_lag_seconds": global_lag_vector,
            "global_pearson_r": global_pearson_r_vector,
            "analysis_start_seconds": np.full(number_of_records, ANALYSIS_START),
            "analysis_end_seconds": analysis_end_vector,
        },
        index=column_labels,
    )
    global_summary_csv = safe_to_csv(
        global_summary_dataframe,
        OUTPUT_DIR / "zero_time_summary_matrix.csv",
        index=True,
    )
    print(f"GLOBAL zero-shift results: {global_csv}")
    print(f"Zero-shift summary: {global_summary_csv}")
    print(f"Bandwise zero-shift Pearson r matrix: {max_pearson_r_csv}")
    print(f"Band time-shift matrix (valid entries must be zero): {best_lag_csv}")
    print(f"Band residual-shift matrix (valid entries must be zero): {delta_lag_csv}")
    program_total_seconds = elapsed_seconds(program_start)
    timing_log.append(
        {
            "record": "ALL",
            "band": "ALL",
            "data_type": "program",
            "stage": "program_total",
            "seconds": program_total_seconds,
        }
    )
    timing_dataframe = pd.DataFrame(timing_log)
    timing_csv = safe_to_csv(
        timing_dataframe, OUTPUT_DIR / "zero_time_runtime_timing_log.csv", index=False
    )
    if not timing_dataframe.empty and "seconds" in timing_dataframe.columns:
        timing_seconds_numeric = pd.to_numeric(
            timing_dataframe["seconds"], errors="coerce"
        )
        valid_mask = timing_seconds_numeric.notna()
        valid_timing = timing_dataframe.loc[valid_mask].copy()
        valid_timing["seconds"] = timing_seconds_numeric.loc[valid_mask].to_numpy(
            dtype=np.float64
        )
        slowest_steps = valid_timing.sort_values(by="seconds", ascending=False).head(15)
        print("\n" + "=" * 88)
        print("The 15 slowest steps")
        print("=" * 88)
        print(
            slowest_steps[
                ["record", "band", "data_type", "stage", "seconds"]
            ].to_string(index=False)
        )
    print("\n" + "=" * 88)
    print("Calculation completed")
    print("=" * 88)
    print(
        f"Total runtime: {program_total_seconds:.3f} s ({program_total_seconds / 60.0:.2f} min)"
    )
    print(f"Detailed results: {detailed_csv}")
    print(f"Timing log: {timing_csv}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
