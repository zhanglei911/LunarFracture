from pathlib import Path
import csv
import numpy as np
from scipy.signal import detrend, fftconvolve, hilbert
from scipy.stats import linregress
from obspy.signal.filter import bandpass
from obspy.signal.util import smooth

# Read the data/ directory beside this script.
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / "data"
OBS_CACHE_DIR = CACHE_ROOT / "observations"
SIM_CACHE_ROOT = CACHE_ROOT / "simulations"

# Analysis outputs are written beside this public script under results/.
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "decay_time"
CACHE_FS = 4.0
TAUD_FMIN = 0.4
TAUD_FMAX = 0.8

# Width = periods / center frequency: boxcar half-window or Gaussian FWHM.
TAUD_SMOOTH_PERIODS = 150.0
TAUD_SMOOTHER = "boxcar"
TAUD_GAUSSIAN_TRUNCATE_SIGMA = 4.0
TMAX_SEARCH_START = 0.0
TMAX_SEARCH_END = 4500.0

# Fit starts this many seconds after the envelope maximum.
TAUD_DELAY_AFTER_TMAX = 200.0

# Require the complete fit window within the available data.
TAUD_FIT_LENGTH = 1000.0

# True uses forward/backward bandpass filtering; False uses causal filtering.
TAUD_ZERO_PHASE = True

# False analyzes displacement; True differentiates both observations and simulations.
CONVERT_DISPLACEMENT_TO_VELOCITY = False

# Simulation-only padding and right-edge taper before envelope extraction.
SIM_LEFT_ZERO_PADDING_SECONDS = 200.0
SIM_RIGHT_TAPER_SECONDS = 10.0
MINIMUM_FIT_SAMPLES = 300
MINIMUM_FIT_VALID_FRACTION = 0.6

# Record identities, filenames, and component/channel settings used by this script.
RECORDS = [
    (
        "2.2°",
        "A14 LM→S14",
        "A14_LM_S14_minus5000_plus8000_DISP_4Hz.npz",
        "S22_local_radial_4Hz.npz",
    ),
    (
        "2.4°",
        "A12 LM→S12",
        "A12_LM_S12_minus5000_plus8000_DISP_4Hz.npz",
        "S24_local_radial_4Hz.npz",
    ),
    (
        "3.0°",
        "A15 LM→S15",
        "A15_LM_S15_minus5000_plus8000_DISP_4Hz.npz",
        "S30_local_radial_4Hz.npz",
    ),
    (
        "3.8°",
        "A14 LM→S12",
        "A14_LM_S12_minus5000_plus8000_DISP_4Hz.npz",
        "S38_local_radial_4Hz.npz",
    ),
    (
        "4.5°",
        "A13 S-IVB→S12",
        "A13_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz",
        "S45_local_radial_4Hz.npz",
    ),
    (
        "5.1°",
        "A16 S-IVB→S12",
        "A16_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz",
        "S51_local_radial_4Hz.npz",
    ),
    (
        "5.8°",
        "A14 S-IVB→S12",
        "A14_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz",
        "S58_local_radial_4Hz.npz",
    ),
    (
        "6.1°",
        "A15 S-IVB→S14",
        "A15_SIVB_S14_minus5000_plus8000_DISP_4Hz.npz",
        "S61_local_radial_4Hz.npz",
    ),
    (
        "9.1°",
        "A16 S-IVB→S14",
        "A16_SIVB_S14_minus5000_plus8000_DISP_4Hz.npz",
        "S91_local_tangential_4Hz.npz",
    ),
    (
        "11.1°",
        "A17 S-IVB→S12",
        "A17_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz",
        "S111_local_radial_4Hz.npz",
    ),
    (
        "11.7°",
        "A15 S-IVB→S12",
        "A15_SIVB_S12_minus5000_plus8000_DISP_4Hz.npz",
        "S117_local_radial_4Hz.npz",
    ),
    (
        "25.4°",
        "A17 LM→S15",
        "A17_LM_S15_minus5000_plus8000_DISP_4Hz.npz",
        "S254_local_radial_4Hz.npz",
    ),
    (
        "28.0°",
        "A17 S-IVB→S16",
        "A17_SIVB_S16_minus5000_plus8000_DISP_4Hz.npz",
        "S280_local_radial_4Hz.npz",
    ),
]

# Simulation cases retained from the original decay-time analysis.
REQUIRED_CASES = ["Case2", "Case8", "Case9", "Case10", "Case11", "Case12"]


def load_cache(path):
    """Load a waveform cache and reconstruct its time axis at the required sampling rate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cache does not exist:\n{path}")
    with np.load(path, allow_pickle=False) as z:
        for key in ("data", "t0", "dt"):
            if key not in z.files:
                raise ValueError(f"{path}\nMissing fields: {key}")
        data = np.asarray(z["data"], dtype=np.float64)
        t0 = float(z["t0"])
        dt = float(z["dt"])
    if data.ndim != 1:
        raise ValueError(f"data must be one-dimensional: {path}")
    fs = 1.0 / dt
    if not np.isclose(fs, CACHE_FS, rtol=1e-07, atol=1e-10):
        raise ValueError(f"Expected {CACHE_FS:g} Hz; actual {fs:.9f} Hz: \n{path}")
    time = t0 + np.arange(len(data), dtype=np.float64) * dt
    return (time, data)


def fill_nans_for_processing(time, data):
    """Temporarily interpolate missing samples for filtering; the caller restores their mask."""
    time = np.asarray(time, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    valid = np.isfinite(data)
    if np.count_nonzero(valid) < 2:
        raise ValueError(" has fewer than 2 valid samples.")
    output = data.copy()
    missing = ~valid
    if np.any(missing):
        output[missing] = np.interp(time[missing], time[valid], data[valid])
    return output


def right_cosine_taper(data, dt, seconds):
    """Apply a cosine taper to the right edge of a simulation waveform."""
    output = np.asarray(data, dtype=np.float64).copy()
    n = int(round(seconds / dt))
    n = min(n, len(output))
    if n <= 1:
        return output
    phase = np.linspace(0.0, np.pi, n)
    taper = 0.5 * (1.0 + np.cos(phase))
    output[-n:] *= taper
    return output


def taud_smoothing_samples(dt):
    """Convert center-frequency periods to samples and seconds for the configured smoother."""
    center_frequency = 0.5 * (TAUD_FMIN + TAUD_FMAX)
    center_period = 1.0 / center_frequency
    smoothing_seconds = TAUD_SMOOTH_PERIODS * center_period
    samples = max(1, int(round(smoothing_seconds / dt)))
    return (samples, smoothing_seconds)


def gaussian_smooth_taud_envelope(
    data, dt, fwhm_seconds, truncate_sigma=TAUD_GAUSSIAN_TRUNCATE_SIGMA
):
    """Apply reflected-padding Gaussian smoothing using FWHM and sigma truncation."""
    data = np.asarray(data, dtype=np.float64)
    if len(data) < 3:
        raise ValueError("Gaussian smoothing requires at least 3 samples.")
    dt = float(dt)
    fwhm_seconds = float(fwhm_seconds)
    truncate_sigma = float(truncate_sigma)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Invalid sample interval for Gaussian smoothing.")
    if not np.isfinite(fwhm_seconds) or fwhm_seconds <= 0.0:
        raise ValueError("Gaussian FWHM must be positive.")
    if not np.isfinite(truncate_sigma) or truncate_sigma <= 0.0:
        raise ValueError("Gaussian truncation in sigma units must be positive.")
    sigma_seconds = fwhm_seconds / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_samples = max(sigma_seconds / dt, 1.0)
    half_width = min(int(np.ceil(truncate_sigma * sigma_samples)), len(data) - 1)
    sample_lag = np.arange(-half_width, half_width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (sample_lag / sigma_samples) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(data, (half_width, half_width), mode="reflect")
    smoothed_padded = fftconvolve(padded, kernel, mode="same")
    smoothed = smoothed_padded[half_width : half_width + len(data)]
    return np.maximum(smoothed, 0.0)


def smooth_taud_envelope(envelope, dt):
    """Smooth using ObsPy boxcar half-window width or Gaussian FWHM."""
    smoothing_samples, smoothing_seconds = taud_smoothing_samples(dt)
    smoother_name = str(TAUD_SMOOTHER).strip().lower()
    if smoother_name in {"boxcar", "moving_average", "obspy"}:
        smoothed = smooth(
            x=np.asarray(envelope, dtype=np.float64), smoothie=smoothing_samples
        )
    elif smoother_name in {"gaussian", "gauss"}:
        smoothed = gaussian_smooth_taud_envelope(
            data=envelope,
            dt=dt,
            fwhm_seconds=smoothing_seconds,
            truncate_sigma=TAUD_GAUSSIAN_TRUNCATE_SIGMA,
        )
    else:
        raise ValueError(
            f"Unknown TAUD_SMOOTHER: {TAUD_SMOOTHER!r}. Choose boxcar or gaussian."
        )
    return (np.asarray(smoothed, dtype=np.float64), smoothing_seconds)


def calculate_taud_envelope(time, displacement, is_simulation):
    """Build the decay-analysis envelope and restore the original missing-sample mask."""
    time = np.asarray(time, dtype=np.float64)
    displacement = np.asarray(displacement, dtype=np.float64)
    if len(time) != len(displacement):
        raise ValueError("time/data lengths differ.")
    dt = float(np.median(np.diff(time)))
    original_missing = ~np.isfinite(displacement)
    x = fill_nans_for_processing(time, displacement)
    x = detrend(x, type="linear")
    if is_simulation:
        x = right_cosine_taper(x, dt, SIM_RIGHT_TAPER_SECONDS)
    pad_samples = 0
    if is_simulation and time[0] > -SIM_LEFT_ZERO_PADDING_SECONDS:
        required_seconds = time[0] + SIM_LEFT_ZERO_PADDING_SECONDS
        pad_samples = max(0, int(round(required_seconds / dt)))
        if pad_samples > 0:
            x = np.r_[np.zeros(pad_samples, dtype=np.float64), x]
    if CONVERT_DISPLACEMENT_TO_VELOCITY:
        x = np.gradient(x, dt)
    filtered = bandpass(
        x,
        freqmin=TAUD_FMIN,
        freqmax=TAUD_FMAX,
        df=1.0 / dt,
        corners=4,
        zerophase=TAUD_ZERO_PHASE,
    )
    envelope = np.abs(hilbert(filtered))
    envelope, smoothing_seconds = smooth_taud_envelope(envelope, dt)
    if pad_samples > 0:
        envelope = envelope[pad_samples:]
    if len(envelope) != len(time):
        raise RuntimeError("Envelope length does not match the original time axis.")
    envelope[original_missing] = np.nan
    return (envelope, smoothing_seconds)


def find_tmax(time, envelope):
    """Find the maximum finite envelope value in the configured Tmax search window."""
    time = np.asarray(time, dtype=np.float64)
    envelope = np.asarray(envelope, dtype=np.float64)
    mask = (
        (time >= TMAX_SEARCH_START) & (time <= TMAX_SEARCH_END) & np.isfinite(envelope)
    )
    if np.count_nonzero(mask) < 10:
        raise ValueError("Too few valid envelope samples within the Tmax search range.")
    t = time[mask]
    e = envelope[mask]
    index = int(np.argmax(e))
    return (float(t[index]), float(e[index]))


def fit_taud(time, envelope):
    """Fit log amplitude over a complete window; require sufficient data and a negative slope."""
    time = np.asarray(time, dtype=np.float64)
    envelope = np.asarray(envelope, dtype=np.float64)
    tmax, emax = find_tmax(time, envelope)
    fit_start = tmax + TAUD_DELAY_AFTER_TMAX
    fit_end = fit_start + TAUD_FIT_LENGTH
    data_start = float(time[0])
    data_end = float(time[-1])
    if fit_start < data_start:
        raise ValueError(
            f"Fit start lies outside the data range: fit_start={fit_start:.1f} s, data_start={data_start:.1f} s"
        )
    if fit_end > data_end:
        raise ValueError(
            f"No complete {TAUD_FIT_LENGTH:.0f} s decay window: Tmax={tmax:.1f} s, required fit end={fit_end:.1f} s, but data end at {data_end:.1f} s"
        )
    window_mask = (time >= fit_start) & (time <= fit_end)
    expected_count = int(np.count_nonzero(window_mask))
    fit_mask = window_mask & np.isfinite(envelope) & (envelope > 0.0)
    valid_count = int(np.count_nonzero(fit_mask))
    if expected_count <= 0:
        raise ValueError("The tau_d fit window contains no samples.")
    valid_fraction = valid_count / expected_count
    if valid_count < MINIMUM_FIT_SAMPLES:
        raise ValueError(f"Number of valid tau_d fit samples is only {valid_count}.")
    if valid_fraction < MINIMUM_FIT_VALID_FRACTION:
        raise ValueError(
            f"Valid fraction in the tau_d fit window is {valid_fraction:.3f}, below {MINIMUM_FIT_VALID_FRACTION:.2f}"
        )
    t_fit = np.asarray(time[fit_mask], dtype=np.float64)
    e_fit = np.asarray(envelope[fit_mask], dtype=np.float64)
    x = t_fit - fit_start
    y = np.log(e_fit)
    regression = linregress(x, y)
    slope = float(regression.slope)
    if not np.isfinite(slope) or slope >= 0.0:
        raise ValueError(f"Invalid decay-fit slope: {slope:.6e}")
    tau_d = -1.0 / slope
    r_squared = float(regression.rvalue**2)
    return {
        "tau_d": float(tau_d),
        "tmax": float(tmax),
        "emax": float(emax),
        "fit_start": float(fit_start),
        "fit_end": float(fit_end),
        "slope": float(slope),
        "r2": float(r_squared),
        "valid_fraction": float(valid_fraction),
        "valid_count": int(valid_count),
    }


def calculate_observation_taud():
    """Calculate decay fits for valid observation records and report skipped failures."""
    results = []
    print("\n" + "=" * 100)
    print("Apollo tau_d")
    print("=" * 100)
    for distance, event, obs_file, sim_file in RECORDS:
        path = OBS_CACHE_DIR / obs_file
        try:
            time, data = load_cache(path)
            envelope, smoothing_seconds = calculate_taud_envelope(
                time, data, is_simulation=False
            )
            fit = fit_taud(time, envelope)
            result = {
                "type": "Apollo",
                "case": "Observation",
                "distance": distance,
                "event": event,
                "smoother": TAUD_SMOOTHER,
                "smoothing_width_definition": (
                    "FWHM"
                    if str(TAUD_SMOOTHER).lower() == "gaussian"
                    else "half-window"
                ),
                "fmin_hz": TAUD_FMIN,
                "fmax_hz": TAUD_FMAX,
                "smooth_periods": TAUD_SMOOTH_PERIODS,
                "delay_after_tmax_s": TAUD_DELAY_AFTER_TMAX,
                "fit_length_s": TAUD_FIT_LENGTH,
                "smoothing_seconds": smoothing_seconds,
                **fit,
            }
            results.append(result)
            print(
                f"{distance:>5s} | {event:<18s} | Tmax={fit['tmax']:7.1f} s | fit={fit['fit_start']:7.1f}–{fit['fit_end']:7.1f} s | tau_d={fit['tau_d']:8.1f} s | R²={fit['r2']:.3f}"
            )
        except Exception as error:
            print(f"{distance:>5s} | {event:<18s} | FAILED: {error}")
    return results


def calculate_simulation_taud():
    """Calculate decay fits for each required simulation case and report skipped failures."""
    all_results = {case_name: [] for case_name in REQUIRED_CASES}
    for case_name in REQUIRED_CASES:
        print("\n" + "=" * 100)
        print(f"{case_name} tau_d")
        print("=" * 100)
        case_directory = SIM_CACHE_ROOT / case_name
        if not case_directory.exists():
            raise FileNotFoundError(f"Case directory does not exist:\n{case_directory}")
        for distance, event, obs_file, sim_file in RECORDS:
            path = case_directory / sim_file
            try:
                time, data = load_cache(path)
                envelope, smoothing_seconds = calculate_taud_envelope(
                    time, data, is_simulation=True
                )
                fit = fit_taud(time, envelope)
                result = {
                    "type": "Simulation",
                    "case": case_name,
                    "distance": distance,
                    "event": event,
                    "smoother": TAUD_SMOOTHER,
                    "smoothing_width_definition": (
                        "FWHM"
                        if str(TAUD_SMOOTHER).lower() == "gaussian"
                        else "half-window"
                    ),
                    "fmin_hz": TAUD_FMIN,
                    "fmax_hz": TAUD_FMAX,
                    "smooth_periods": TAUD_SMOOTH_PERIODS,
                    "delay_after_tmax_s": TAUD_DELAY_AFTER_TMAX,
                    "fit_length_s": TAUD_FIT_LENGTH,
                    "smoothing_seconds": smoothing_seconds,
                    **fit,
                }
                all_results[case_name].append(result)
                print(
                    f"{distance:>5s} | Tmax={fit['tmax']:7.1f} s | fit={fit['fit_start']:7.1f}–{fit['fit_end']:7.1f} s | tau_d={fit['tau_d']:8.1f} s | R²={fit['r2']:.3f}"
                )
            except Exception as error:
                print(f"{distance:>5s} | FAILED: {error}")
    return all_results


def save_taud_csv(obs_results, sim_results):
    """Export successful observation and simulation fits with the analysis settings."""
    output_path = OUTPUT_DIR / "tau_d_all_records.csv"
    rows = []
    rows.extend(obs_results)
    for case_name in REQUIRED_CASES:
        rows.extend(sim_results[case_name])
    fields = [
        "type",
        "case",
        "distance",
        "event",
        "smoother",
        "smoothing_width_definition",
        "fmin_hz",
        "fmax_hz",
        "smooth_periods",
        "delay_after_tmax_s",
        "fit_length_s",
        "smoothing_seconds",
        "tau_d",
        "tmax",
        "emax",
        "fit_start",
        "fit_end",
        "slope",
        "r2",
        "valid_fraction",
        "valid_count",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    print(f"\nDetailed tau_d CSV:\n{output_path}")


def main():
    """Run the configured processing and write numerical outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    width_samples, width_seconds = taud_smoothing_samples(1.0 / CACHE_FS)
    width_definition = (
        "FWHM"
        if str(TAUD_SMOOTHER).strip().lower() in {"gaussian", "gauss"}
        else "half-window"
    )
    print(
        f"Smoothing width: {width_seconds:g} s ({width_samples} samples; {width_definition})"
    )
    print(
        f"Simulation left padding: {SIM_LEFT_ZERO_PADDING_SECONDS:g} s; right taper: {SIM_RIGHT_TAPER_SECONDS:g} s"
    )
    print(f"Cache sampling rate: {CACHE_FS:g} Hz")
    print(f"Band: {TAUD_FMIN:g}-{TAUD_FMAX:g} Hz; zero phase: {TAUD_ZERO_PHASE}")
    print(f"Smoother: {TAUD_SMOOTHER}; periods: {TAUD_SMOOTH_PERIODS:g}")
    print(f"Convert displacement to velocity: {CONVERT_DISPLACEMENT_TO_VELOCITY}")
    print(f"Peak search: {TMAX_SEARCH_START:g}-{TMAX_SEARCH_END:g} s")
    print(f"Fit delay: {TAUD_DELAY_AFTER_TMAX:g} s; length: {TAUD_FIT_LENGTH:g} s")
    print(
        f"Minimum fit samples: {MINIMUM_FIT_SAMPLES}; valid fraction: {MINIMUM_FIT_VALID_FRACTION:g}"
    )
    print(f"Cases: {', '.join(REQUIRED_CASES)}; records: {len(RECORDS)}")
    obs_results = calculate_observation_taud()
    sim_results = calculate_simulation_taud()
    save_taud_csv(obs_results, sim_results)


if __name__ == "__main__":
    main()
