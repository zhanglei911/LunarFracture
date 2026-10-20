from __future__ import annotations
import csv
import re
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy.signal import resample_poly

# Raw SPECFEM2D input, when available, and generated caches live in data/.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
ALL_CASES_DIR = DATA_ROOT / "AllCases"
CACHE_ROOT = DATA_ROOT
SIM_CACHE_ROOT = CACHE_ROOT / "simulations"
SUMMARY_CSV = SIM_CACHE_ROOT / "simulation_preprocess_summary.csv"

# For an independent comparison, point this at a separate reference cache tree.
REFERENCE_BUILDER_SIM_CACHE_ROOT = SIM_CACHE_ROOT
TARGET_FS = 4.0
TARGET_DT = 1.0 / TARGET_FS
COMPRESS_NPZ = True

# When False, preserve any existing cache file.
OVERWRITE = False

# Use None to process every discovered case; otherwise process these names.
CASE_NAME_FILTER = [f"Case{i}" for i in range(1, 13)]

# Enable only after selecting a separate, independent reference cache tree.
VERIFY_AGAINST_BUILDER_CACHE = False
ARC_CENTER_X = 0.0
ARC_CENTER_Z = 0.0
VERTICAL_SIGN = 1.0
TANGENTIAL_SIGN = 1.0

# Record identities, filenames, and component/channel settings used by this script.
RECORDS = [
    {
        "distance": "2.2°",
        "event_station": "A14 LM→S14",
        "sim_x_file": "SF.S22.BXX.semd",
        "sim_z_file": "SF.S22.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": -310352.45,
        "station_z": 1709151.18,
        "component": "radial",
    },
    {
        "distance": "2.4°",
        "event_station": "A12 LM→S12",
        "sim_x_file": "SF.S24.BXX.semd",
        "sim_z_file": "SF.S24.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": -304384.51,
        "station_z": 1710224.1,
        "component": "radial",
    },
    {
        "distance": "3.0°",
        "event_station": "A15 LM→S15",
        "sim_x_file": "SF.S30.BXX.semd",
        "sim_z_file": "SF.S30.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": -286458.72,
        "station_z": 1713317.78,
        "component": "radial",
    },
    {
        "distance": "3.8°",
        "event_station": "A14 LM→S12",
        "sim_x_file": "SF.S38.BXX.semd",
        "sim_z_file": "SF.S38.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": -262509.14,
        "station_z": 1717150.36,
        "component": "radial",
    },
    {
        "distance": "4.5°",
        "event_station": "A13 S-IVB→S12",
        "sim_x_file": "SF.S45.BXX.semd",
        "sim_z_file": "SF.S45.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": -241511.13,
        "station_z": 1720229.28,
        "component": "radial",
    },
    {
        "distance": "5.1°",
        "event_station": "A16 S-IVB→S12",
        "sim_x_file": "SF.S51.BXX.semd",
        "sim_z_file": "SF.S51.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": -223484.01,
        "station_z": 1722664.01,
        "component": "radial",
    },
    {
        "distance": "5.8°",
        "event_station": "A14 S-IVB→S12",
        "sim_x_file": "SF.S58.BXX.semd",
        "sim_z_file": "SF.S58.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": -202421.55,
        "station_z": 1725265.76,
        "component": "radial",
    },
    {
        "distance": "6.1°",
        "event_station": "A15 S-IVB→S14",
        "sim_x_file": "SF.S61.BXX.semd",
        "sim_z_file": "SF.S61.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": -193385.34,
        "station_z": 1726301.98,
        "component": "radial",
    },
    {
        "distance": "9.1°",
        "event_station": "A16 S-IVB→S14",
        "sim_x_file": "SF.S91.BXX.semd",
        "sim_z_file": "SF.S91.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": -102772.65,
        "station_z": 1734057.15,
        "component": "tangential",
    },
    {
        "distance": "11.1°",
        "event_station": "A17 S-IVB→S12",
        "sim_x_file": "SF.S111.BXX.semd",
        "sim_z_file": "SF.S111.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": -42192.32,
        "station_z": 1736587.52,
        "component": "radial",
    },
    {
        "distance": "11.7°",
        "event_station": "A15 S-IVB→S12",
        "sim_x_file": "SF.S117.BXX.semd",
        "sim_z_file": "SF.S117.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": -24004.84,
        "station_z": 1736934.13,
        "component": "radial",
    },
    {
        "distance": "25.4°",
        "event_station": "A17 LM→S15",
        "sim_x_file": "SF.S254.BXX.semd",
        "sim_z_file": "SF.S254.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": 388050.38,
        "station_z": 1693202.09,
        "component": "radial",
    },
    {
        "distance": "28.0°",
        "event_station": "A17 S-IVB→S16",
        "sim_x_file": "SF.S280.BXX.semd",
        "sim_z_file": "SF.S280.BXZ.semd",
        "sim_source_time": 0.0,
        "station_x": 464459.61,
        "station_z": 1673855.93,
        "component": "radial",
    },
]


def natural_case_key(path: Path):
    """Return a numeric-aware sorting key for case directory names."""
    name = path.name
    match = re.match("(?i)^case(\\d+)(.*)$", name)
    if match:
        return (0, int(match.group(1)), match.group(2).lower(), name.lower())
    return (1, 10**9, name.lower(), name.lower())


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


def record_label(record: dict) -> str:
    """Build the record label used by cache metadata and manual exclusions."""
    return f"{record['distance']} {record['event_station']}"


def station_code_from_sim_filename(filename: str) -> str:
    """Extract the station code from a SPECFEM2D component filename."""
    match = re.search("\\.(S\\d+)\\.", str(filename), flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Cannot extract the station code from the simulation filename: {filename}"
        )
    return match.group(1).upper()


def selected_case_dirs() -> List[Path]:
    """Find and naturally sort case directories selected by CASE_NAME_FILTER."""
    if not ALL_CASES_DIR.exists():
        raise FileNotFoundError(f"ALL_CASES_DIR does not exist: {ALL_CASES_DIR}")
    case_dirs = [
        path
        for path in ALL_CASES_DIR.iterdir()
        if path.is_dir() and path.name.lower().startswith("case")
    ]
    if CASE_NAME_FILTER is not None:
        wanted = {str(x).lower() for x in CASE_NAME_FILTER}
        case_dirs = [path for path in case_dirs if path.name.lower() in wanted]
    case_dirs.sort(key=natural_case_key)
    return case_dirs


def find_simulation_file(case_dir: Path, filename: str) -> Path:
    """Locate a configured simulation component within a case directory."""
    case_dir = Path(case_dir)
    preferred = [
        case_dir / filename,
        case_dir / "OUTPUT_FILES" / filename,
        case_dir / "output_files" / filename,
        case_dir / "OUTPUT" / filename,
        case_dir / "output" / filename,
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    matches = list(case_dir.rglob(filename))
    if len(matches) == 0:
        raise FileNotFoundError(f"{case_dir.name} does not contain: {filename}")
    matches.sort(key=lambda p: (len(p.relative_to(case_dir).parts), str(p).lower()))
    if len(matches) > 1:
        print(
            f"    Warning: {case_dir.name} contains multiple matches for {filename}; using: {matches[0]}"
        )
    return matches[0]


def clean_and_regularize(time, data):
    """Drop invalid pairs, sort and deduplicate times, and interpolate irregular sampling."""
    time = np.asarray(time, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(data)
    time = time[valid]
    data = data[valid]
    if len(time) < 20:
        raise ValueError("Fewer than 20 valid data samples.")
    order = np.argsort(time)
    time = time[order]
    data = data[order]
    unique = np.concatenate(([True], np.diff(time) > 0))
    time = time[unique]
    data = data[unique]
    if len(time) < 20:
        raise ValueError("Too few samples remain after deduplication.")
    dt_array = np.diff(time)
    dt = np.median(dt_array)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time sample interval.")
    maximum_error = np.max(np.abs(dt_array - dt))
    if maximum_error > dt * 0.001:
        regular_time = np.arange(time[0], time[-1] + 0.5 * dt, dt)
        regular_data = np.interp(regular_time, time, data)
        time = regular_time
        data = regular_data
    return (time, data)


def load_semd_component(file_path):
    """Load time and the last column of a SEM displacement component."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Simulation component file does not exist: {file_path}"
        )
    array = np.loadtxt(file_path, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(
            f"Invalid simulation file format; at least two columns are required: {file_path}"
        )
    return clean_and_regularize(array[:, 0], array[:, -1])


def calculate_local_basis_2d(station_x, station_z, center_x, center_z):
    """Construct outward radial and orthogonal tangential unit vectors in the X/Z plane."""
    if station_x is None or station_z is None:
        raise ValueError("station_x or station_z is missing.")
    station_x = float(station_x)
    station_z = float(station_z)
    center_x = float(center_x)
    center_z = float(center_z)
    radial_x = station_x - center_x
    radial_z = station_z - center_z
    radius = np.hypot(radial_x, radial_z)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(
            "Station coincides with the arc center or has invalid coordinates."
        )
    er_x = radial_x / radius
    er_z = radial_z / radius
    et_x = -er_z
    et_z = er_x
    angle_from_x_deg = np.degrees(np.arctan2(er_z, er_x))
    angle_from_z_deg = np.degrees(np.arctan2(er_x, er_z))
    return {
        "station_x": station_x,
        "station_z": station_z,
        "center_x": center_x,
        "center_z": center_z,
        "radius": radius,
        "er_x": er_x,
        "er_z": er_z,
        "et_x": et_x,
        "et_z": et_z,
        "angle_from_x_deg": angle_from_x_deg,
        "angle_from_z_deg": angle_from_z_deg,
    }


def load_and_rotate_simulation(
    x_component_file,
    z_component_file,
    station_x,
    station_z,
    center_x,
    center_z,
    source_time=0.0,
    component="radial",
):
    """Align X/Z components, rotate into the configured basis, and subtract source time."""
    time_x, ux = load_semd_component(x_component_file)
    time_z, uz = load_semd_component(z_component_file)
    same_time_axis = len(time_x) == len(time_z) and np.allclose(
        time_x, time_z, rtol=1e-07, atol=1e-09
    )
    if same_time_axis:
        common_time = time_x
        ux_common = ux
        uz_common = uz
    else:
        common_start = max(time_x[0], time_z[0])
        common_end = min(time_x[-1], time_z[-1])
        if common_end <= common_start:
            raise ValueError("BXX and BXZ have no overlapping time range.")
        common_dt = max(np.median(np.diff(time_x)), np.median(np.diff(time_z)))
        common_time = np.arange(common_start, common_end + 0.5 * common_dt, common_dt)
        ux_common = np.interp(common_time, time_x, ux)
        uz_common = np.interp(common_time, time_z, uz)
    geometry = calculate_local_basis_2d(
        station_x=station_x, station_z=station_z, center_x=center_x, center_z=center_z
    )
    component = str(component).strip().lower()
    if component == "radial":
        rotated = VERTICAL_SIGN * (
            ux_common * geometry["er_x"] + uz_common * geometry["er_z"]
        )
        component_name = "local_radial"
        kind = "simulation_local_radial"
    elif component == "tangential":
        rotated = TANGENTIAL_SIGN * (
            ux_common * geometry["et_x"] + uz_common * geometry["et_z"]
        )
        component_name = "local_tangential"
        kind = "simulation_local_tangential"
    else:
        raise ValueError("component must be 'radial' or 'tangential'.")
    common_time = common_time - float(source_time)
    return (
        common_time,
        np.asarray(rotated, dtype=np.float64),
        geometry,
        component_name,
        kind,
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


def compare_with_builder_reference(
    case_dir: Path,
    station_code: str,
    component_name: str,
    new_time: np.ndarray,
    new_data: np.ndarray,
):
    """Compare radial output with an available configured reference cache."""
    result = {
        "reference_cache_exists": False,
        "reference_time_equal": "",
        "reference_data_equal": "",
        "reference_max_abs_diff": "",
        "reference_rms_diff": "",
    }
    if not VERIFY_AGAINST_BUILDER_CACHE or component_name != "local_radial":
        return result
    reference_path = (
        REFERENCE_BUILDER_SIM_CACHE_ROOT
        / case_dir.name
        / f"{station_code}_local_radial_{TARGET_FS:g}Hz.npz"
    )
    if not reference_path.exists():
        return result
    result["reference_cache_exists"] = True
    ref_time, ref_data, _ = load_4hz_cache(reference_path)
    time_equal = len(ref_time) == len(new_time) and np.array_equal(ref_time, new_time)
    result["reference_time_equal"] = bool(time_equal)
    if len(ref_data) != len(new_data):
        result["reference_data_equal"] = False
        print("        VERIFY: Different lengths; pointwise comparison is unavailable.")
        return result
    same_nan = np.array_equal(np.isnan(ref_data), np.isnan(new_data))
    finite = np.isfinite(ref_data) & np.isfinite(new_data)
    if np.any(finite):
        difference = new_data[finite] - ref_data[finite]
        max_abs_diff = float(np.max(np.abs(difference)))
        rms_diff = float(np.sqrt(np.mean(difference**2)))
    else:
        max_abs_diff = np.nan
        rms_diff = np.nan
    exact_data_equal = same_nan and np.array_equal(ref_data[finite], new_data[finite])
    result["reference_data_equal"] = bool(exact_data_equal)
    result["reference_max_abs_diff"] = max_abs_diff
    result["reference_rms_diff"] = rms_diff
    print(
        f"        VERIFY builder cache: time_equal={time_equal}, data_exact={exact_data_equal}, max|diff|={max_abs_diff:.3e}, RMS(diff)={rms_diff:.3e}"
    )
    return result


def preprocess_one_record(case_dir: Path, record: dict):
    """Preprocess one configured record, save its cache, and return summary metadata."""
    label = record_label(record)
    x_path = find_simulation_file(case_dir, record["sim_x_file"])
    z_path = find_simulation_file(case_dir, record["sim_z_file"])
    station_code = station_code_from_sim_filename(record["sim_z_file"])
    simulation_time, simulation_data, geometry, component_name, kind = (
        load_and_rotate_simulation(
            x_component_file=x_path,
            z_component_file=z_path,
            station_x=record["station_x"],
            station_z=record["station_z"],
            center_x=ARC_CENTER_X,
            center_z=ARC_CENTER_Z,
            source_time=record["sim_source_time"],
            component=record["component"],
        )
    )
    time_4hz, data_4hz, rs_info = resample_gap_aware(
        simulation_time, simulation_data, target_fs=TARGET_FS
    )
    case_output_dir = SIM_CACHE_ROOT / case_dir.name
    case_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        case_output_dir / f"{station_code}_{component_name}_{TARGET_FS:g}Hz.npz"
    )
    save_dict = {
        "data": data_4hz.astype(np.float32),
        "t0": np.float64(time_4hz[0]),
        "dt": np.float64(TARGET_DT),
        "fs": np.float64(TARGET_FS),
        "kind": np.array(kind),
        "case_name": np.array(case_dir.name),
        "record_label": np.array(label),
        "distance": np.array(str(record["distance"])),
        "event_station": np.array(str(record["event_station"])),
        "station_code": np.array(station_code),
        "source_x_file": np.array(str(x_path)),
        "source_z_file": np.array(str(z_path)),
        "source_fs": np.float64(rs_info["source_fs"]),
        "source_dt": np.float64(rs_info["source_dt"]),
        "resample_up": np.int64(rs_info["resample_up"]),
        "resample_down": np.int64(rs_info["resample_down"]),
        "station_x": np.float64(geometry["station_x"]),
        "station_z": np.float64(geometry["station_z"]),
        "er_x": np.float64(geometry["er_x"]),
        "er_z": np.float64(geometry["er_z"]),
    }
    if component_name == "local_radial":
        save_dict["vertical_sign"] = np.float64(VERTICAL_SIGN)
    else:
        save_dict["et_x"] = np.float64(geometry["et_x"])
        save_dict["et_z"] = np.float64(geometry["et_z"])
        save_dict["tangential_sign"] = np.float64(TANGENTIAL_SIGN)
    if output_path.exists() and (not OVERWRITE):
        print(f"        File exists; skipping save: {output_path}")
        saved_time, saved_data, _ = load_4hz_cache(output_path)
        verification = compare_with_builder_reference(
            case_dir=case_dir,
            station_code=station_code,
            component_name=component_name,
            new_time=saved_time,
            new_data=saved_data,
        )
    else:
        choose_npz_save(output_path, **save_dict)
        verification = compare_with_builder_reference(
            case_dir=case_dir,
            station_code=station_code,
            component_name=component_name,
            new_time=time_4hz,
            new_data=np.asarray(data_4hz, dtype=np.float32).astype(np.float64),
        )
    print(f"        component = {component_name}")
    print(f"        e_r = ({geometry['er_x']:+.8f}, {geometry['er_z']:+.8f})")
    print(f"        e_t = ({geometry['et_x']:+.8f}, {geometry['et_z']:+.8f})")
    print(
        f"        {rs_info['source_fs']:.6f} Hz -> {TARGET_FS:.3f} Hz, npts {rs_info['input_npts']:,} -> {rs_info['output_npts']:,}"
    )
    print(f"        t0={time_4hz[0]:+.6f} s, dt={TARGET_DT:.6f} s")
    print(f"        cache = {output_path}")
    row = {
        "case": case_dir.name,
        "distance": record["distance"],
        "event_station": record["event_station"],
        "station_code": station_code,
        "component": component_name,
        "kind": kind,
        "station_x": geometry["station_x"],
        "station_z": geometry["station_z"],
        "er_x": geometry["er_x"],
        "er_z": geometry["er_z"],
        "et_x": geometry["et_x"],
        "et_z": geometry["et_z"],
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
        "output_path": str(output_path),
        **verification,
        "status": "OK",
        "error": "",
    }
    return row


SUMMARY_FIELDS = [
    "case",
    "distance",
    "event_station",
    "station_code",
    "component",
    "kind",
    "station_x",
    "station_z",
    "er_x",
    "er_z",
    "et_x",
    "et_z",
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
    "output_path",
    "reference_cache_exists",
    "reference_time_equal",
    "reference_data_equal",
    "reference_max_abs_diff",
    "reference_rms_diff",
    "status",
    "error",
]


def save_summary(rows):
    """Write processing summary rows to SUMMARY_CSV."""
    SIM_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run the configured processing and write numerical outputs."""
    print(
        f"Case filter: {CASE_NAME_FILTER}; overwrite caches: {OVERWRITE}; compress NPZ: {COMPRESS_NPZ}"
    )
    print(
        f"Arc center: ({ARC_CENTER_X:g}, {ARC_CENTER_Z:g}); radial sign: {VERTICAL_SIGN:g}; tangential sign: {TANGENTIAL_SIGN:g}"
    )
    print(f"Reference comparison enabled: {VERIFY_AGAINST_BUILDER_CACHE}")
    if (
        VERIFY_AGAINST_BUILDER_CACHE
        and REFERENCE_BUILDER_SIM_CACHE_ROOT.resolve() == SIM_CACHE_ROOT.resolve()
    ):
        print(
            "Reference and output roots coincide; comparison is not an independent validation."
        )
    print(
        "Selected components: "
        + "; ".join((f"{record_label(r)}: {r['component']}" for r in RECORDS))
    )
    case_dirs = selected_case_dirs()
    if len(case_dirs) == 0:
        raise RuntimeError("No cases were selected.")
    print("=" * 100)
    print("Standalone SPECFEM2D preprocessing - exact builder pipeline")
    print("=" * 100)
    print(f"Cases root     = {ALL_CASES_DIR}")
    print(f"Output root    = {SIM_CACHE_ROOT}")
    print(f"Reference root = {REFERENCE_BUILDER_SIM_CACHE_ROOT}")
    print(f"Target fs      = {TARGET_FS:.3f} Hz")
    print(f"Record count   = {len(RECORDS)}")
    print("\nNumerical pipeline:")
    print(
        "  loadtxt(last column) -> clean_and_regularize -> X/Z align -> rotate -> source_time -> builder resample_gap_aware -> builder-style NPZ"
    )
    rows = []
    for case_index, case_dir in enumerate(case_dirs, start=1):
        print("\n" + "=" * 100)
        print(f"[CASE {case_index}/{len(case_dirs)}] {case_dir.name}")
        print("=" * 100)
        for record_index, record in enumerate(RECORDS, start=1):
            print(
                f"\n    [{record_index:02d}/{len(RECORDS):02d}] {record_label(record)}"
            )
            try:
                row = preprocess_one_record(case_dir, record)
            except Exception as error:
                print(f"        FAILED: {error}")
                row = {field: "" for field in SUMMARY_FIELDS}
                row.update(
                    {
                        "case": case_dir.name,
                        "distance": record["distance"],
                        "event_station": record["event_station"],
                        "component": (
                            "local_tangential"
                            if record["component"] == "tangential"
                            else "local_radial"
                        ),
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
    print("  reference_time_equal")
    print("  reference_data_equal")
    print("  reference_max_abs_diff")
    print("\nFor available radial reference caches, exact agreement is expected:")
    print("  reference_time_equal = True")
    print("  reference_data_equal = True")
    print("  reference_max_abs_diff = 0")
    print("\nTangential components are not compared with radial reference caches.")


if __name__ == "__main__":
    main()
