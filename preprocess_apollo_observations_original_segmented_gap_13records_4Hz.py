from __future__ import annotations
import csv
import sys
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from obspy import read, read_inventory, UTCDateTime, Stream
from scipy.signal import resample_poly

# Inputs and generated caches are stored in data/ beside this script.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
LONG_MSEED_DIR = DATA_ROOT / "original Apollo data" / "all_impacts_minus5000_plus8000"
INVENTORY_FILE = DATA_ROOT / "inventory.xml"
CACHE_ROOT = DATA_ROOT
OBS_CACHE_DIR = CACHE_ROOT / "observations"
SUMMARY_CSV = OBS_CACHE_DIR / "observation_preprocess_summary.csv"
# Use the PDART copy shipped beside this script, even when input data live elsewhere.
PDART_PARENT_DIR = Path(__file__).resolve().parent
PDART_UTIL = PDART_PARENT_DIR / "pdart" / "util.py"
if not PDART_UTIL.exists():
    raise FileNotFoundError(
        f"Apollo PDART was not found:\n{PDART_UTIL}\n\nThis file provides single-sample gap interpolation and ATT timing correction."
    )
if str(PDART_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PDART_PARENT_DIR))
try:
    from pdart.util import linear_interpolation, timing_correction
except ImportError as error:
    raise ImportError(
        "Found pdart/util.py but could not import it.\nInstall its dependencies in the active Python environment."
    ) from error
PDART_INTERPOLATION_LIMIT = 1
ENABLE_PDART_TIMING_CORRECTION = True
REQUIRE_ATT_FOR_TIMING_CORRECTION = True

# The downstream cache format and analyses require displacement (DISP).
RESPONSE_OUTPUT = "DISP"

# Instrument-response prefilter corners in Hz; distinct from analysis bandpass settings.
RESPONSE_PRE_FILT = (0.03, 0.05, 0.9, 1.1)
RESPONSE_WATER_LEVEL = None

# Each response-removal taper is limited by both the fraction and duration below.
RESPONSE_TAPER_MAX_FRACTION = 0.05
RESPONSE_TAPER_MAX_SECONDS = 20.0
TARGET_FS = 4.0
TARGET_DT = 1.0 / TARGET_FS
COMPRESS_NPZ = True

# When False, preserve any existing cache file.
OVERWRITE = False

# Inclusive impact-relative intervals (seconds); these values define manual masking.
MANUAL_EXCLUDE_WINDOWS = {
    "25.4° A17 LM→S15": [(4100.0, 4170.0, "suspected transient near 4.1 ks")],
    "2.4° A12 LM→S12": [
        (3739.0, 3844.0, "suspected broadband transient"),
        (4147.0, 4300.0, "suspected broadband transient"),
    ],
}

# Record identities, filenames, and component/channel settings used by this script.
RECORDS = [
    {
        "distance": "2.2°",
        "event_station": "A14 LM→S14",
        "impact_time": "1971-02-07T00:45:25.7",
        "station": "S14",
        "channel": "MHZ",
        "mseed_file": "A14_LM_S14_minus5000_plus8000.mseed",
    },
    {
        "distance": "2.4°",
        "event_station": "A12 LM→S12",
        "impact_time": "1969-11-20T22:17:17.7",
        "station": "S12",
        "channel": "MHZ",
        "mseed_file": "A12_LM_S12_minus5000_plus8000.mseed",
    },
    {
        "distance": "3.0°",
        "event_station": "A15 LM→S15",
        "impact_time": "1971-08-03T03:03:37.0",
        "station": "S15",
        "channel": "MHZ",
        "mseed_file": "A15_LM_S15_minus5000_plus8000.mseed",
    },
    {
        "distance": "3.8°",
        "event_station": "A14 LM→S12",
        "impact_time": "1971-02-07T00:45:25.7",
        "station": "S12",
        "channel": "MHZ",
        "mseed_file": "A14_LM_S12_minus5000_plus8000.mseed",
    },
    {
        "distance": "4.5°",
        "event_station": "A13 S-IVB→S12",
        "impact_time": "1970-04-15T01:09:41.0",
        "station": "S12",
        "channel": "MHZ",
        "mseed_file": "A13_SIVB_S12_minus5000_plus8000.mseed",
    },
    {
        "distance": "5.1°",
        "event_station": "A16 S-IVB→S12",
        "impact_time": "1972-04-19T21:02:03.0",
        "station": "S12",
        "channel": "MHZ",
        "mseed_file": "A16_SIVB_S12_minus5000_plus8000.mseed",
    },
    {
        "distance": "5.8°",
        "event_station": "A14 S-IVB→S12",
        "impact_time": "1971-02-04T07:40:55.4",
        "station": "S12",
        "channel": "MHZ",
        "mseed_file": "A14_SIVB_S12_minus5000_plus8000.mseed",
    },
    {
        "distance": "6.1°",
        "event_station": "A15 S-IVB→S14",
        "impact_time": "1971-07-29T20:58:42.9",
        "station": "S14",
        "channel": "MHZ",
        "mseed_file": "A15_SIVB_S14_minus5000_plus8000.mseed",
    },
    {
        "distance": "9.1°",
        "event_station": "A16 S-IVB→S14",
        "impact_time": "1972-04-19T21:02:03.0",
        "station": "S14",
        "channel": "MH1",
        "mseed_file": "A16_SIVB_S14_minus5000_plus8000.mseed",
    },
    {
        "distance": "11.1°",
        "event_station": "A17 S-IVB→S12",
        "impact_time": "1972-12-10T20:32:42.3",
        "station": "S12",
        "channel": "MHZ",
        "mseed_file": "A17_SIVB_S12_minus5000_plus8000.mseed",
    },
    {
        "distance": "11.7°",
        "event_station": "A15 S-IVB→S12",
        "impact_time": "1971-07-29T20:58:42.9",
        "station": "S12",
        "channel": "MHZ",
        "mseed_file": "A15_SIVB_S12_minus5000_plus8000.mseed",
    },
    {
        "distance": "25.4°",
        "event_station": "A17 LM→S15",
        "impact_time": "1972-12-15T06:50:20.8",
        "station": "S15",
        "channel": "MHZ",
        "mseed_file": "A17_LM_S15_minus5000_plus8000.mseed",
    },
    {
        "distance": "28.0°",
        "event_station": "A17 S-IVB→S16",
        "impact_time": "1972-12-10T20:32:42.3",
        "station": "S16",
        "channel": "MHZ",
        "mseed_file": "A17_SIVB_S16_minus5000_plus8000.mseed",
    },
]


def record_label(record: dict) -> str:
    """Build the record label used by cache metadata and manual exclusions."""
    return f"{record['distance']} {record['event_station']}"


def finite_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return half-open [start, end) index pairs for contiguous True runs."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional.")
    if len(mask) == 0:
        return []
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts, ends))


def choose_npz_save(path: Path, **arrays):
    """Save an NPZ cache using the configured compression setting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if COMPRESS_NPZ:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def load_4hz_cache(path: Path):
    """Load cache arrays and metadata and reconstruct the time axis."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        data = np.asarray(z["data"], dtype=np.float64)
        t0 = float(z["t0"])
        dt = float(z["dt"])
        meta = {}
        for key in z.files:
            if key in {"data", "t0", "dt"}:
                continue
            value = z[key]
            if np.ndim(value) == 0:
                try:
                    value = value.item()
                except Exception:
                    pass
            meta[key] = value
    time = t0 + np.arange(len(data), dtype=np.float64) * dt
    return (time, data, meta)


def validate_response_pre_filt(sampling_rate, pre_filt):
    """Require ordered positive response-filter corners below the native Nyquist frequency."""
    nyquist = 0.5 * sampling_rate
    f1, f2, f3, f4 = pre_filt
    if not 0.0 < f1 < f2 < f3 < f4 < nyquist:
        raise ValueError(
            f"Response pre_filt must satisfy 0<f1<f2<f3<f4<Nyquist.\nsampling_rate={sampling_rate:.6f} Hz, Nyquist={nyquist:.6f} Hz, pre_filt={pre_filt}"
        )


def convert_trace_data_to_float64_preserve_mask(trace):
    """Convert trace data to float64 without discarding an existing missing-data mask."""
    data = trace.data
    if np.ma.isMaskedArray(data):
        mask = np.ma.getmaskarray(data).copy()
        values = np.asarray(np.ma.getdata(data), dtype=np.float64)
        trace.data = np.ma.array(values, mask=mask, copy=False)
    else:
        trace.data = np.asarray(data, dtype=np.float64)
    return trace


def interpolate_att_float_safe(trace, interpolation_limit=1):
    """Interpolate short ATT gaps after converting timestamp data to float64."""
    data = np.ma.asarray(trace.data)
    values = np.asarray(np.ma.getdata(data), dtype=np.float64).copy()
    mask = np.ma.getmaskarray(data).copy()
    mask |= ~np.isfinite(values)
    mask |= values == -1.0
    values[mask] = np.nan
    series = pd.Series(values, dtype="float64")
    series.interpolate(
        method="linear",
        axis=0,
        limit=int(interpolation_limit),
        limit_direction=None,
        limit_area="inside",
        inplace=True,
    )
    trace.data = series.to_numpy(dtype=np.float64)
    return trace


def prepare_att_stream_for_timing_correction(station_att, interpolation_limit=1):
    """Prepare an ATT stream without filling long timestamp gaps."""
    prepared = station_att.copy()
    for trace in prepared:
        interpolate_att_float_safe(trace, interpolation_limit=interpolation_limit)
    prepared.merge(method=0, fill_value=np.nan)
    for trace in prepared:
        trace.data = np.asarray(
            np.ma.filled(np.ma.asarray(trace.data, dtype=np.float64), np.nan),
            dtype=np.float64,
        )
    return prepared


def validate_att_covers_correction_time(att_stream, correction_time, station):
    """Check that valid ATT timestamps cover the requested correction time."""
    values = []
    total_samples = 0
    missing_samples = 0
    for trace in att_stream:
        data = np.asarray(
            np.ma.filled(np.ma.asarray(trace.data, dtype=np.float64), np.nan),
            dtype=np.float64,
        )
        total_samples += int(data.size)
        valid = np.isfinite(data) & (data != -1.0)
        missing_samples += int(np.count_nonzero(~valid))
        if np.any(valid):
            values.append(data[valid])
    if not values:
        raise ValueError(
            f"Station {station} ATT contains no valid timestamps; total samples={total_samples}; missing/invalid={missing_samples}."
        )
    all_values = np.concatenate(values)
    target = float(UTCDateTime(correction_time).timestamp)
    minimum = float(np.min(all_values))
    maximum = float(np.max(all_values))
    if not minimum <= target <= maximum:
        raise ValueError(
            f"Station {station} ATT does not cover the correction time {UTCDateTime(correction_time)}; ATT range={UTCDateTime(minimum)}–{UTCDateTime(maximum)}."
        )


def build_regular_axis_with_nan_gaps(corrected_segments, impact_time):
    """Map corrected segments onto one impact-relative axis and retain missing data as NaN."""
    if len(corrected_segments) == 0:
        raise ValueError(
            "No continuous observation segments are available to construct the time axis."
        )
    sampling_rates = np.array(
        [float(trace.stats.sampling_rate) for trace in corrected_segments],
        dtype=np.float64,
    )
    reference_sampling_rate = float(np.median(sampling_rates))
    if not np.allclose(sampling_rates, reference_sampling_rate, rtol=1e-07, atol=1e-10):
        raise ValueError(
            f"Continuous segments of the same observation have inconsistent sampling rates: {sampling_rates.tolist()}"
        )
    dt = 1.0 / reference_sampling_rate
    impact_time = UTCDateTime(impact_time)
    relative_starts = np.array(
        [float(trace.stats.starttime - impact_time) for trace in corrected_segments],
        dtype=np.float64,
    )
    relative_ends = np.array(
        [float(trace.stats.endtime - impact_time) for trace in corrected_segments],
        dtype=np.float64,
    )
    full_start = float(np.min(relative_starts))
    full_end = float(np.max(relative_ends))
    full_time = np.arange(full_start, full_end + 0.5 * dt, dt, dtype=np.float64)
    full_data = np.full(full_time.shape, np.nan, dtype=np.float64)
    for trace in corrected_segments:
        segment_data = np.asarray(
            np.ma.filled(np.ma.asarray(trace.data, dtype=np.float64), np.nan),
            dtype=np.float64,
        )
        segment_start = float(trace.stats.starttime - impact_time)
        start_index = int(round((segment_start - full_start) / dt))
        end_index = start_index + len(segment_data)
        if start_index < 0 or end_index > len(full_data):
            raise RuntimeError("Continuous segment maps outside the common time axis.")
        destination = full_data[start_index:end_index]
        source_valid = np.isfinite(segment_data)
        destination[source_valid] = segment_data[source_valid]
        full_data[start_index:end_index] = destination
    return (full_time, full_data, reference_sampling_rate)


def apply_manual_exclusion_windows(time, data, label):
    """Replace configured inclusive time windows with NaN and report exclusion counts."""
    time = np.asarray(time, dtype=np.float64)
    output = np.asarray(data, dtype=np.float64).copy()
    windows = MANUAL_EXCLUDE_WINDOWS.get(str(label), [])
    total_masked = 0
    applied = []
    for item in windows:
        if len(item) == 2:
            start_s, end_s = item
            reason = "manual exclusion"
        else:
            start_s, end_s, reason = item
        start_s = float(start_s)
        end_s = float(end_s)
        if end_s <= start_s:
            raise ValueError(
                f"Invalid manual exclusion window: {label} {start_s}–{end_s} s"
            )
        mask = (time >= start_s) & (time <= end_s) & np.isfinite(output)
        masked_here = int(np.count_nonzero(mask))
        output[mask] = np.nan
        total_masked += masked_here
        applied.append((start_s, end_s, str(reason), masked_here))
    if applied:
        print(f"    Manual exclusion: {label}; count={total_masked} valid samples.")
        for start_s, end_s, reason, count in applied:
            print(
                f"      {start_s:.1f}–{end_s:.1f} s, {count} samples; reason: {reason}"
            )
    return (
        output,
        {
            "manual_exclusion_window_count": float(len(applied)),
            "manual_excluded_sample_count": float(total_masked),
            "manual_exclusion_windows": str(applied),
        },
    )


def load_corrected_observation_channel(
    file_path, impact_time, inventory, station, channel, label
):
    """Apply ATT timing correction, segmented response removal, and manual exclusions."""
    file_path = Path(file_path)
    impact_time = UTCDateTime(impact_time)
    station = str(station)
    channel = str(channel)
    if not file_path.exists():
        raise FileNotFoundError(f"Observation file does not exist: {file_path}")
    stream = read(str(file_path))
    seismic_original = stream.select(station=station, channel=channel).copy()
    if len(seismic_original) == 0:
        available = sorted(
            {trace.stats.channel for trace in stream if trace.stats.station == station}
        )
        raise ValueError(
            f"{file_path.name} does not contain {station}.{channel}.\nAvailable channels at this station: {available}"
        )
    original_gaps = seismic_original.get_gaps()
    corrected_seismic = Stream()
    timing_shifts = []
    att_trace_count = 0
    station_codes = sorted({trace.stats.station for trace in seismic_original})
    for station_code in station_codes:
        station_seismic = seismic_original.select(station=station_code).copy()
        station_att = stream.select(station=station_code, channel="ATT").copy()
        if len(station_att) == 0:
            message = f"{file_path.name}: station {station_code} has no ATT channel; PDART timing_correction cannot be applied."
            if REQUIRE_ATT_FOR_TIMING_CORRECTION:
                raise ValueError(message)
            print(f"    Warning: {message}")
            corrected_seismic += station_seismic
            continue
        att_trace_count += len(station_att)
        station_seismic_prepared = station_seismic.copy()
        for trace in station_seismic_prepared:
            convert_trace_data_to_float64_preserve_mask(trace)
            linear_interpolation(trace, interpolation_limit=PDART_INTERPOLATION_LIMIT)
            convert_trace_data_to_float64_preserve_mask(trace)
        station_att_prepared = prepare_att_stream_for_timing_correction(
            station_att, interpolation_limit=PDART_INTERPOLATION_LIMIT
        )
        station_work = station_seismic_prepared + station_att_prepared
        before_seismic = station_work.select(station=station_code, channel=channel)
        if len(before_seismic) == 0:
            raise RuntimeError(
                f"Cannot find the channel before timing correction: {station_code}.{channel}."
            )
        before_start = min((trace.stats.starttime for trace in before_seismic))
        if ENABLE_PDART_TIMING_CORRECTION:
            validate_att_covers_correction_time(
                station_work.select(channel="ATT"), impact_time, station_code
            )
            try:
                timing_correction(
                    station_work, correction_time=impact_time, interpolate=False
                )
            except Exception as error:
                raise RuntimeError(
                    f"Station {station_code} failed PDART timing_correction.\nOriginal error: {error}"
                ) from error
        after_seismic = station_work.select(station=station_code, channel=channel)
        if len(after_seismic) == 0:
            raise RuntimeError(
                f"No channel remains after timing correction: {station_code}.{channel}."
            )
        after_start = min((trace.stats.starttime for trace in after_seismic))
        timing_shifts.append(float(after_start - before_start))
        corrected_seismic += after_seismic.copy()
    if len(corrected_seismic) == 0:
        raise ValueError("No target seismic channel remains after PDART processing.")
    corrected_seismic.merge(method=0, fill_value=None)
    remaining_gaps = corrected_seismic.get_gaps()
    continuous_segments = corrected_seismic.split()
    if len(continuous_segments) == 0:
        raise ValueError(
            "No continuous valid segments remain after PDART timing correction and merging."
        )
    print(f"    Selected channel = {channel}")
    print(f"    PDART ATT traces = {att_trace_count}")
    print(
        f"    timing shift median = {np.median(timing_shifts):+.6f} s"
        if timing_shifts
        else "    timing shift median = NaN"
    )
    print(
        f"    original gaps = {len(original_gaps)}, remaining gaps = {len(remaining_gaps)}, segments = {len(continuous_segments)}"
    )
    corrected_segments = []
    for segment_index, trace in enumerate(continuous_segments, start=1):
        trace = trace.copy()
        sampling_rate = float(trace.stats.sampling_rate)
        validate_response_pre_filt(sampling_rate, RESPONSE_PRE_FILT)
        try:
            inventory.get_response(trace.id, trace.stats.starttime)
        except Exception as error:
            raise ValueError(
                f"Inventory has no response for {trace.id} at {trace.stats.starttime}.\nOriginal error: {error}"
            ) from error
        segment_duration_seconds = max(
            float(trace.stats.endtime - trace.stats.starttime), 1.0 / sampling_rate
        )
        response_taper_fraction = min(
            float(RESPONSE_TAPER_MAX_FRACTION),
            float(RESPONSE_TAPER_MAX_SECONDS) / segment_duration_seconds,
        )
        response_taper_fraction = float(np.clip(response_taper_fraction, 0.0, 0.5))
        trace.remove_response(
            inventory=inventory,
            output=RESPONSE_OUTPUT,
            pre_filt=RESPONSE_PRE_FILT,
            water_level=RESPONSE_WATER_LEVEL,
            zero_mean=True,
            taper=True,
            taper_fraction=response_taper_fraction,
            plot=False,
        )
        corrected_segments.append(trace)
    observation_time, observation_data, sampling_rate = (
        build_regular_axis_with_nan_gaps(
            corrected_segments=corrected_segments, impact_time=impact_time
        )
    )
    finite_before_manual = np.isfinite(observation_data)
    if np.count_nonzero(finite_before_manual) < 100:
        raise ValueError(
            "Fewer than 100 valid observation samples remain after response removal."
        )
    if np.nanstd(observation_data) <= 0:
        raise ValueError("Response-corrected observation has zero amplitude variation.")
    observation_data, manual_qc = apply_manual_exclusion_windows(
        observation_time, observation_data, label
    )
    return (
        np.asarray(observation_time, dtype=np.float64),
        np.asarray(observation_data, dtype=np.float64),
        {
            "selected_channel": channel,
            "station": station,
            "source_fs": float(sampling_rate),
            "original_gap_count": int(len(original_gaps)),
            "remaining_gap_count": int(len(remaining_gaps)),
            "continuous_segment_count": int(len(continuous_segments)),
            "att_trace_count": int(att_trace_count),
            "timing_shift_seconds_median": (
                float(np.median(timing_shifts)) if timing_shifts else np.nan
            ),
            "timing_shift_seconds_minimum": (
                float(np.min(timing_shifts)) if timing_shifts else np.nan
            ),
            "timing_shift_seconds_maximum": (
                float(np.max(timing_shifts)) if timing_shifts else np.nan
            ),
            "manual_qc": manual_qc,
        },
    )


def target_time_axis(source_time: np.ndarray, target_fs: float) -> np.ndarray:
    """Construct a target-rate time axis aligned to integer multiples of its interval."""
    source_time = np.asarray(source_time, dtype=np.float64)
    if len(source_time) < 2:
        raise ValueError("Too few original time samples.")
    start = float(source_time[0])
    end = float(source_time[-1])
    start_tick = int(np.ceil(start * target_fs - 1e-10))
    end_tick = int(np.floor(end * target_fs + 1e-10))
    if end_tick <= start_tick:
        raise ValueError("The resampled time range is too short.")
    ticks = np.arange(start_tick, end_tick + 1, dtype=np.int64)
    return ticks.astype(np.float64) / float(target_fs)


def resample_gap_aware(
    source_time: np.ndarray, source_data: np.ndarray, target_fs: float = TARGET_FS
):
    """Polyphase-resample each finite segment and retain gaps on the target time axis."""
    source_time = np.asarray(source_time, dtype=np.float64)
    source_data = np.asarray(source_data, dtype=np.float64)
    if (
        source_time.ndim != 1
        or source_data.ndim != 1
        or len(source_time) != len(source_data)
    ):
        raise ValueError(
            "source_time/source_data must be one-dimensional and have equal lengths."
        )
    if len(source_time) < 3:
        raise ValueError("Too few source samples.")
    if not np.all(np.isfinite(source_time)):
        raise ValueError("source_time contains NaN/Inf.")
    if not np.all(np.diff(source_time) > 0.0):
        raise ValueError("source_time must be strictly increasing.")
    source_dt = float(np.median(np.diff(source_time)))
    source_fs = 1.0 / source_dt
    if target_fs <= 0.0:
        raise ValueError("target_fs must be positive.")
    if target_fs > source_fs * (1.0 + 1e-08):
        raise ValueError(
            f"This script requires downsampling: source fs={source_fs:.6f} Hz; target fs={target_fs:.6f} Hz."
        )
    target_time = target_time_axis(source_time=source_time, target_fs=target_fs)
    target_data = np.full(target_time.shape, np.nan, dtype=np.float64)
    valid_mask = np.isfinite(source_data)
    runs = finite_runs(valid_mask)
    if len(runs) == 0:
        raise ValueError("No finite data are available for resampling.")
    ratio = Fraction(float(target_fs / source_fs)).limit_denominator(100000)
    up = int(ratio.numerator)
    down = int(ratio.denominator)
    actual_output_fs = source_fs * up / down
    relative_fs_error = abs(actual_output_fs - target_fs) / target_fs
    if relative_fs_error > 1e-06:
        raise ValueError(
            f"Cannot represent the sampling-rate ratio with sufficient accuracy: source_fs={source_fs:.9f}, target_fs={target_fs:.9f}, actual={actual_output_fs:.9f}"
        )
    processed_runs = 0
    for start_index, end_index in runs:
        segment_time = source_time[start_index:end_index]
        segment_data = source_data[start_index:end_index]
        if len(segment_data) < 8:
            continue
        segment_resampled = resample_poly(
            segment_data, up=up, down=down, padtype="line"
        )
        segment_resampled_time = (
            segment_time[0]
            + np.arange(len(segment_resampled), dtype=np.float64) / actual_output_fs
        )
        usable = segment_resampled_time <= segment_time[-1] + 1e-10
        segment_resampled_time = segment_resampled_time[usable]
        segment_resampled = segment_resampled[usable]
        if len(segment_resampled) < 2:
            continue
        inside_target = (target_time >= segment_resampled_time[0]) & (
            target_time <= segment_resampled_time[-1]
        )
        if not np.any(inside_target):
            continue
        target_data[inside_target] = np.interp(
            target_time[inside_target], segment_resampled_time, segment_resampled
        )
        processed_runs += 1
    if processed_runs == 0:
        raise ValueError("All continuous segments are too short to resample.")
    info = {
        "source_dt": source_dt,
        "source_fs": source_fs,
        "target_dt": 1.0 / target_fs,
        "target_fs": target_fs,
        "resample_up": up,
        "resample_down": down,
        "actual_output_fs": actual_output_fs,
        "input_npts": len(source_data),
        "output_npts": len(target_data),
        "input_valid_fraction": float(np.mean(valid_mask)),
        "output_valid_fraction": float(np.mean(np.isfinite(target_data))),
        "input_segment_count": len(runs),
        "processed_segment_count": processed_runs,
    }
    return (target_time, target_data, info)


def preprocess_one_record(record, inventory):
    """Preprocess one configured record, save its cache, and return summary metadata."""
    label = record_label(record)
    source_path = LONG_MSEED_DIR / record["mseed_file"]
    impact_time = UTCDateTime(record["impact_time"])
    station = record["station"]
    channel = record["channel"]
    print(f"    source  = {source_path}")
    print(f"    station = {station}")
    print(f"    channel = {channel}")
    print(f"    impact  = {impact_time}")
    observation_time, observation_data, obs_info = load_corrected_observation_channel(
        file_path=source_path,
        impact_time=impact_time,
        inventory=inventory,
        station=station,
        channel=channel,
        label=label,
    )
    time_4hz, data_4hz, rs_info = resample_gap_aware(
        observation_time, observation_data, target_fs=TARGET_FS
    )
    output_path = OBS_CACHE_DIR / f"{source_path.stem}_DISP_{TARGET_FS:g}Hz.npz"
    if output_path.exists() and (not OVERWRITE):
        print(f"    File exists; not overwriting: {output_path}")
    else:
        choose_npz_save(
            output_path,
            data=data_4hz.astype(np.float32),
            t0=np.float64(time_4hz[0]),
            dt=np.float64(TARGET_DT),
            fs=np.float64(TARGET_FS),
            kind=np.array("observation_DISP"),
            record_label=np.array(label),
            distance=np.array(str(record["distance"])),
            event_station=np.array(str(record["event_station"])),
            station=np.array(str(station)),
            selected_channel=np.array(str(channel)),
            impact_time=np.array(str(impact_time)),
            source_file=np.array(str(source_path)),
            source_fs=np.float64(rs_info["source_fs"]),
            source_dt=np.float64(rs_info["source_dt"]),
            resample_up=np.int64(rs_info["resample_up"]),
            resample_down=np.int64(rs_info["resample_down"]),
            valid_fraction=np.float64(rs_info["output_valid_fraction"]),
            timing_shift_seconds_median=np.float64(
                obs_info["timing_shift_seconds_median"]
            ),
            manual_excluded_sample_count=np.float64(
                obs_info["manual_qc"].get("manual_excluded_sample_count", 0.0)
            ),
        )
    print(f"    native fs = {rs_info['source_fs']:.6f} Hz")
    print(f"    {TARGET_FS:g} Hz      = {time_4hz[0]:+.3f} to {time_4hz[-1]:+.3f} s")
    print(f"    valid     = {rs_info['output_valid_fraction']:.6f}")
    print(f"    cache     = {output_path}")
    manual_qc = obs_info["manual_qc"]
    return {
        "distance": record["distance"],
        "event_station": record["event_station"],
        "station": station,
        "selected_channel": channel,
        "impact_time": str(impact_time),
        "source_file": str(source_path),
        "source_fs": rs_info["source_fs"],
        "source_dt": rs_info["source_dt"],
        "target_fs": TARGET_FS,
        "target_dt": TARGET_DT,
        "resample_up": rs_info["resample_up"],
        "resample_down": rs_info["resample_down"],
        "input_npts": rs_info["input_npts"],
        "output_npts": rs_info["output_npts"],
        "input_valid_fraction": rs_info["input_valid_fraction"],
        "output_valid_fraction": rs_info["output_valid_fraction"],
        "original_gap_count": obs_info["original_gap_count"],
        "remaining_gap_count": obs_info["remaining_gap_count"],
        "continuous_segment_count": obs_info["continuous_segment_count"],
        "att_trace_count": obs_info["att_trace_count"],
        "timing_shift_seconds_median": obs_info["timing_shift_seconds_median"],
        "timing_shift_seconds_minimum": obs_info["timing_shift_seconds_minimum"],
        "timing_shift_seconds_maximum": obs_info["timing_shift_seconds_maximum"],
        "manual_exclusion_window_count": manual_qc.get(
            "manual_exclusion_window_count", 0.0
        ),
        "manual_excluded_sample_count": manual_qc.get(
            "manual_excluded_sample_count", 0.0
        ),
        "manual_exclusion_windows": manual_qc.get("manual_exclusion_windows", "[]"),
        "output_path": str(output_path),
        "status": "OK",
        "error": "",
    }


SUMMARY_FIELDS = [
    "distance",
    "event_station",
    "station",
    "selected_channel",
    "impact_time",
    "source_file",
    "source_fs",
    "source_dt",
    "target_fs",
    "target_dt",
    "resample_up",
    "resample_down",
    "input_npts",
    "output_npts",
    "input_valid_fraction",
    "output_valid_fraction",
    "original_gap_count",
    "remaining_gap_count",
    "continuous_segment_count",
    "att_trace_count",
    "timing_shift_seconds_median",
    "timing_shift_seconds_minimum",
    "timing_shift_seconds_maximum",
    "manual_exclusion_window_count",
    "manual_excluded_sample_count",
    "manual_exclusion_windows",
    "output_path",
    "status",
    "error",
]


def save_summary(rows):
    """Write processing summary rows to SUMMARY_CSV."""
    OBS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run the configured processing and write numerical outputs."""
    print(
        f"ATT timing correction: {ENABLE_PDART_TIMING_CORRECTION}; ATT required: {REQUIRE_ATT_FOR_TIMING_CORRECTION}"
    )
    print(f"PDART interpolation limit: {PDART_INTERPOLATION_LIMIT}")
    print(
        f"Response output: {RESPONSE_OUTPUT}; prefilter: {RESPONSE_PRE_FILT}; water level: {RESPONSE_WATER_LEVEL}"
    )
    print(
        f"Response taper limits: fraction={RESPONSE_TAPER_MAX_FRACTION:g}, duration={RESPONSE_TAPER_MAX_SECONDS:g} s"
    )
    print(f"Manual exclusion windows (seconds): {MANUAL_EXCLUDE_WINDOWS}")
    print(f"Overwrite caches: {OVERWRITE}; compress NPZ: {COMPRESS_NPZ}")
    print(
        "Selected channels: "
        + "; ".join((f"{record_label(r)}: {r['channel']}" for r in RECORDS))
    )
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(f"Inventory file was not found:\n{INVENTORY_FILE}")
    if not LONG_MSEED_DIR.exists():
        raise FileNotFoundError(
            f"Long MSEED directory was not found:\n{LONG_MSEED_DIR}"
        )
    inventory = read_inventory(str(INVENTORY_FILE))
    OBS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("Apollo observation preprocessing - exact builder/v6 base pipeline")
    print("=" * 100)
    print(f"Long MSEED = {LONG_MSEED_DIR}")
    print(f"Inventory  = {INVENTORY_FILE}")
    print(f"Output     = {OBS_CACHE_DIR}")
    print(f"Target fs  = {TARGET_FS:.3f} Hz")
    print(f"Records    = {len(RECORDS)}")
    rows = []
    for index, record in enumerate(RECORDS, start=1):
        label = record_label(record)
        print("\n" + "=" * 100)
        print(f"[OBS {index:02d}/{len(RECORDS):02d}] {label}")
        print("=" * 100)
        try:
            row = preprocess_one_record(record, inventory)
        except Exception as error:
            print(f"    FAILED: {error}")
            row = {field: "" for field in SUMMARY_FIELDS}
            row.update(
                {
                    "distance": record["distance"],
                    "event_station": record["event_station"],
                    "station": record["station"],
                    "selected_channel": record["channel"],
                    "impact_time": record["impact_time"],
                    "source_file": str(LONG_MSEED_DIR / record["mseed_file"]),
                    "target_fs": TARGET_FS,
                    "target_dt": TARGET_DT,
                    "status": "FAILED",
                    "error": repr(error),
                }
            )
        rows.append(row)
        save_summary(rows)
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Summary:\n{SUMMARY_CSV}")
    print("\nCheck these summary fields:")
    print("  selected_channel")
    print("  timing_shift_seconds_median")
    print("  output_valid_fraction")
    print("  manual_excluded_sample_count")


if __name__ == "__main__":
    main()
