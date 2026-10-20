import argparse
import csv
from pathlib import Path

import numpy as np


# Preserve the Case and band order used for the original similarity calculation.
CASE_ORDER = (
    "Case1",
    "Case2",
    "Case3",
    "Case4",
    "Case5",
    "Case6",
    "Case7",
    "Case8",
    "Case9",
    "Case10",
    "Case11",
    "Case12",
)
BAND_ORDER = ("LP", "MP1", "MP2", "HP1", "HP2", "HP3", "global")
EXPECTED_DISTANCE_COLUMNS = 13

DEFAULT_INPUT = (
    Path(__file__).absolute().parent
    / "all_case_pearson_matrices.npz"
)


def decode_names(values):
    """Decode NPZ string or byte labels while preserving their original order."""
    names = [value.decode("utf-8") if isinstance(value, bytes) else str(value)
             for value in values]
    if len(names) != len(set(names)):
        raise ValueError("The NPZ contains duplicate Case or band names.")
    return names


def load_ordered_matrices(input_path):
    """Load and validate the NPZ, then apply the source script's ordering."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input NPZ does not exist: {input_path}")

    with np.load(input_path, allow_pickle=False) as data:
        required = {"case_names", "band_names", "matrices"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Input NPZ is missing fields: {sorted(missing)}")
        case_names = decode_names(data["case_names"])
        band_names = decode_names(data["band_names"])
        matrices = np.asarray(data["matrices"], dtype=np.float64)

    if matrices.ndim != 3:
        raise ValueError("matrices must have Case, band, and distance axes.")
    if matrices.shape[:2] != (len(case_names), len(band_names)):
        raise ValueError("Matrix dimensions do not match the Case/band labels.")
    if matrices.shape[2] != EXPECTED_DISTANCE_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_DISTANCE_COLUMNS} distance columns; "
            f"found {matrices.shape[2]}."
        )
    if not np.all(np.isfinite(matrices)):
        raise ValueError("matrices contains NaN or infinity.")

    missing_cases = sorted(set(CASE_ORDER).difference(case_names))
    missing_bands = sorted(set(BAND_ORDER).difference(band_names))
    if missing_cases or missing_bands:
        raise ValueError(
            f"Missing configured Cases: {missing_cases}; "
            f"missing configured bands: {missing_bands}"
        )

    case_indices = [case_names.index(name) for name in CASE_ORDER]
    band_indices = [band_names.index(name) for name in BAND_ORDER]
    ordered = matrices[np.ix_(case_indices, band_indices,
                              range(EXPECTED_DISTANCE_COLUMNS))]
    return ordered


def calculate_case_similarity(matrices):
    """Return symmetric RMSE and clipped 1-RMSE matrices for all Case pairs."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[0] == 0:
        raise ValueError("Expected a nonempty [Case, band, distance] array.")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("matrices contains NaN or infinity.")

    n_cases = matrices.shape[0]
    rmse_matrix = np.zeros((n_cases, n_cases), dtype=np.float64)
    similarity_matrix = np.zeros((n_cases, n_cases), dtype=np.float64)

    for i in range(n_cases):
        for j in range(i, n_cases):
            # Match matrix_rmse() and calculate_rmse_similarity() in the source.
            rmse = float(np.sqrt(np.mean((matrices[i] - matrices[j]) ** 2)))
            similarity = float(np.clip(1.0 - rmse, 0.0, 1.0))
            rmse_matrix[i, j] = rmse_matrix[j, i] = rmse
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    return rmse_matrix, similarity_matrix


def write_matrix_csv(path, case_names, values):
    """Write a labeled square matrix using full-precision numeric strings."""
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["case", *case_names])
        for name, row in zip(case_names, values):
            writer.writerow([name, *(format(float(value), ".17g") for value in row)])


def write_pairs_csv(path, case_names, rmse_matrix, similarity_matrix):
    """Write each unique Case pair once, excluding self comparisons."""
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["case_a", "case_b", "rmse", "similarity_1_minus_rmse"])
        for i, case_a in enumerate(case_names):
            for j in range(i + 1, len(case_names)):
                writer.writerow(
                    [case_a, case_names[j],
                     format(float(rmse_matrix[i, j]), ".17g"),
                     format(float(similarity_matrix[i, j]), ".17g")]
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="NPZ containing case_names, band_names, and matrices",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: case_similarity_results beside the input)",
    )
    args = parser.parse_args(argv)

    matrices = load_ordered_matrices(args.input)
    rmse_matrix, similarity_matrix = calculate_case_similarity(matrices)
    output_dir = (args.output_dir or args.input.absolute().parent /
                  "case_similarity_results").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    rmse_path = output_dir / "case_rmse_matrix.csv"
    similarity_path = output_dir / "case_similarity_1_minus_rmse_matrix.csv"
    pairs_path = output_dir / "case_similarity_pairs.csv"
    write_matrix_csv(rmse_path, CASE_ORDER, rmse_matrix)
    write_matrix_csv(similarity_path, CASE_ORDER, similarity_matrix)
    write_pairs_csv(pairs_path, CASE_ORDER, rmse_matrix, similarity_matrix)

    print(f"Input: {args.input.absolute()}")
    print(f"Compared {len(CASE_ORDER)} Cases across {len(BAND_ORDER)} bands "
          f"and {EXPECTED_DISTANCE_COLUMNS} distance columns.")
    print(f"RMSE matrix: {rmse_path}")
    print(f"1-RMSE similarity matrix: {similarity_path}")
    print(f"Unique Case pairs: {pairs_path}")


if __name__ == "__main__":
    main()
