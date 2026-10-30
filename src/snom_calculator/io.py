"""CSV inputs and outputs used by the Streamlit workspace."""

from __future__ import annotations

import csv
import io
import json
import re
from typing import BinaryIO, Mapping

import numpy as np

from .models import SpectrumResult


def _data_lines(text: str) -> list[str]:
    """Strip comments and leading key/value metadata, preserving table headers."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "%", "//")):
            continue
        if not lines and re.match(r"^[A-Za-z][^,;\t]*?\s*[:=]", stripped):
            continue
        lines.append(line)
    if not lines:
        raise ValueError("The data file is empty.")
    return lines


def _write_metadata(output: io.StringIO, metadata: Mapping[str, object]) -> None:
    def encode(value: object) -> object:
        if isinstance(value, complex):
            return {"real": value.real, "imag": value.imag}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported metadata value: {type(value).__name__}")

    for key, value in metadata.items():
        rendered = value if isinstance(value, str) else json.dumps(value, default=encode, ensure_ascii=False)
        for line in f"{key}: {rendered}".splitlines():
            output.write(f"# {line}\n")


def parse_experiment(file: BinaryIO) -> dict[str, np.ndarray | None]:
    try:
        table = read_numeric_table(file)
    except Exception as exc:
        raise ValueError("Could not read the experimental CSV/TXT file.") from exc
    names = set(table.dtype.names or ())
    if not {"frequency_cm1", "amplitude"} <= names:
        raise ValueError("Experimental data requires frequency_cm1 and amplitude columns.")
    frequency = np.atleast_1d(table["frequency_cm1"])
    amplitude = np.atleast_1d(table["amplitude"])
    if frequency.size < 2 or not np.all(np.diff(frequency) > 0):
        raise ValueError("Experimental frequency_cm1 values must be increasing.")
    if not np.isfinite(frequency).all() or not np.isfinite(amplitude).all() or np.any(amplitude < 0):
        raise ValueError("Experimental data must be finite; amplitude cannot be negative.")
    phase = np.atleast_1d(table["phase_rad"]) if "phase_rad" in names else None
    if phase is not None and not np.isfinite(phase).all():
        raise ValueError("Experimental phase must be finite.")
    return {"frequency_cm1": frequency, "amplitude": amplitude, "phase_rad": phase}


def read_experiment_columns(file: BinaryIO) -> tuple[list[str], list[list[str]]]:
    """Read a delimited text table without assuming its column names or value types."""
    raw = file.read()
    try:
        content = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as exc:
        raise ValueError("The data file must be UTF-8 text.") from exc
    lines = _data_lines(content)
    delimiter = next((char for char in (",", ";", "\t") if char in lines[0]), None)
    rows = [
        [cell.strip() for cell in next(csv.reader([line], delimiter=delimiter))]
        if delimiter else line.split()
        for line in lines
    ]
    width = len(rows[0])
    if width < 2 or any(len(row) != width for row in rows):
        raise ValueError("Every data row must have the same number of columns (at least two).")
    try:
        # A cell without a digit ("i", "j") parses as a number but is a header; nan/inf stay
        # data so the finite-value check rejects them instead of dropping the row.
        if not all(any(char.isdigit() for char in cell)
                   or cell.lower().lstrip("+-") in {"nan", "inf", "infinity"} for cell in rows[0]):
            raise ValueError
        [complex(cell.replace("i", "j")) for cell in rows[0]]
    except ValueError:
        headers = rows.pop(0)
    else:
        headers = [f"Column {index + 1}" for index in range(width)]
    if not rows:
        raise ValueError("The data file has no numeric rows.")
    return headers, rows


def map_experiment_columns(
    rows: list[list[str]], frequency_col: int, mode: str, signal_cols: tuple[int, ...]
) -> dict[str, np.ndarray | None]:
    if len({frequency_col, *signal_cols}) != 1 + len(signal_cols):
        raise ValueError("Choose a different column for each field.")
    try:
        frequency = np.array([float(row[frequency_col]) for row in rows])
        if mode == "Complex":
            values = np.array([complex(row[signal_cols[0]].replace("i", "j")) for row in rows])
            amplitude, phase = np.abs(values), np.angle(values)
        elif mode == "Real + imaginary":
            values = np.array([complex(float(row[signal_cols[0]]), float(row[signal_cols[1]])) for row in rows])
            amplitude, phase = np.abs(values), np.angle(values)
        elif mode == "Amplitude + phase":
            amplitude = np.array([float(row[signal_cols[0]]) for row in rows])
            phase = np.array([float(row[signal_cols[1]]) for row in rows]) if len(signal_cols) > 1 else None
        else:
            raise ValueError("Unknown signal format.")
    except (IndexError, ValueError) as exc:
        raise ValueError("Selected columns must contain valid numeric values.") from exc
    if frequency.size < 2 or not np.isfinite(frequency).all() or not np.all(np.diff(frequency) > 0):
        raise ValueError("Frequency must contain at least two finite, increasing values.")
    if not np.isfinite(amplitude).all() or np.any(amplitude < 0) or (phase is not None and not np.isfinite(phase).all()):
        raise ValueError("Signal values must be finite; amplitude cannot be negative.")
    return {"frequency_cm1": frequency, "amplitude": amplitude, "phase_rad": phase}


def adjust_experimental_phase(
    frequency_cm1: np.ndarray, phase_rad: np.ndarray,
    offset_rad: float = 0.0, tilt_rad_per_cm1: float = 0.0, *, unwrap: bool = True,
) -> np.ndarray:
    """Rotate phase without changing amplitude; retain full turns when unwrapped."""
    rotation = offset_rad + tilt_rad_per_cm1 * (frequency_cm1 - frequency_cm1[0])
    rotated = np.exp(1j * phase_rad) * np.exp(1j * rotation)
    # Unwrap the source before adding the known rotation: unwrapping the rotated
    # samples would lose full turns for large offsets or steep slopes.
    return np.unwrap(phase_rad) + rotation if unwrap else np.angle(rotated)


def read_numeric_table(file: BinaryIO, *, headerless_columns: Mapping[int, tuple[str, ...]] | None = None) -> np.ndarray:
    raw = file.read()
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    lines = _data_lines(text)
    header = lines[0]
    delimiter = "," if "," in header else None
    names = True
    if headerless_columns is not None:
        cells = header.split(",") if delimiter else header.split()
        try:
            [float(cell) for cell in cells]
        except ValueError:
            pass
        else:
            if len(cells) not in headerless_columns:
                raise ValueError("Unexpected number of columns in headerless table.")
            names = headerless_columns[len(cells)]
    return np.genfromtxt(
        io.StringIO("\n".join(lines)),
        delimiter=delimiter,
        names=names,
        dtype=float,
    )


def result_csv(result: SpectrumResult) -> str:
    output = io.StringIO()
    _write_metadata(output, result.metadata)
    labels = [
        re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        for label in result.permittivities
    ]
    header = [
        "frequency_cm1",
        "sample_harmonic_real",
        "sample_harmonic_imag",
        "reference_harmonic_real",
        "reference_harmonic_imag",
        "contrast_real",
        "contrast_imag",
        "amplitude",
        "phase_rad",
    ]
    for label in labels:
        header.extend([f"{label}_epsilon_real", f"{label}_epsilon_imag"])
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    epsilon_values = list(result.permittivities.values())
    for index, frequency in enumerate(result.frequency_cm1):
        row = [
            frequency,
            result.sample_harmonic[index].real,
            result.sample_harmonic[index].imag,
            result.reference_harmonic[index].real,
            result.reference_harmonic[index].imag,
            result.contrast[index].real,
            result.contrast[index].imag,
            result.amplitude[index],
            result.phase_rad[index],
        ]
        for epsilon in epsilon_values:
            row.extend([epsilon[index].real, epsilon[index].imag])
        writer.writerow(row)
    return output.getvalue()


def combined_results_csv(results: dict[int, SpectrumResult]) -> str:
    """CSV with one frequency column and O{n}A/O{n}P columns per harmonic."""
    output = io.StringIO()
    harmonics = sorted(results)
    _write_metadata(output, {
        **{key: value for key, value in results[harmonics[0]].metadata.items()
           if key != "harmonic"},
        "harmonics": harmonics,
        "harmonic_metadata": {
            str(n): {key: value for key, value in result.metadata.items()
                     if key not in {"tip", "tip_material", "illumination", "numerical", "sample_structure",
                                    "reference_material", "references"}}
            for n, result in sorted(results.items())
        },
    })
    header = ["wn"]
    for harmonic in harmonics:
        header.extend([f"O{harmonic}A", f"O{harmonic}P"])
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    frequency = results[harmonics[0]].frequency_cm1
    for index, freq in enumerate(frequency):
        row = [freq]
        for harmonic in harmonics:
            result = results[harmonic]
            row.extend([result.amplitude[index], result.phase_rad[index]])
        writer.writerow(row)
    return output.getvalue()
