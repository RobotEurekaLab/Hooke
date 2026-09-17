"""Public analysis consumes measured curves, never a simulation mass field."""

import numpy as np
from scipy.optimize import least_squares


def encoder_curve(record):
    rows = np.asarray(record["time_position"], dtype=float)
    if (
        not record.get("valid")
        or rows.ndim != 2
        or rows.shape[1] != 2
        or len(rows) < 30
        or not np.isfinite(rows).all()
    ):
        raise ValueError("Valid encoder observations are required")
    if np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("Encoder timestamps must increase")
    return rows


def oscillation(record):
    rows = encoder_curve(record)
    rows = rows[rows[:, 0] >= 0.2]
    t, x = rows[:, 0], rows[:, 1]
    t = t - t[0]
    if len(t) < 60 or np.ptp(x) < 20 * record["encoder_noise_std_m"]:
        raise ValueError("Oscillation is too short or below the noise floor")
    frequency = np.fft.rfftfreq(len(t), float(np.median(np.diff(t))))
    spectrum = abs(np.fft.rfft(x - x.mean()))
    initial_omega = 2 * np.pi * frequency[1 + np.argmax(spectrum[1:])]
    scale = max(float(np.ptp(x)), 1e-5)

    def prediction(p):
        a, b, omega, decay, offset = p
        return (
            np.exp(-decay * t) * (a * np.cos(omega * t) + b * np.sin(omega * t))
            + offset
        )

    result = least_squares(
        lambda p: (prediction(p) - x) / scale,
        [scale / 2, 0, initial_omega, 0.03, x.mean()],
        bounds=(
            [-scale * 2, -scale * 2, initial_omega * 0.6, 0, -0.03],
            [scale * 2, scale * 2, initial_omega * 1.4, 3, 0.03],
        ),
        max_nfev=1000,
    )
    residual_std = float(np.std(prediction(result.x) - x, ddof=5))
    a, b, omega, decay, _ = result.x
    cycles = float(omega * t[-1] / (2 * np.pi))
    if (
        not result.success
        or cycles < 4
        or np.hypot(a, b) < 10 * residual_std
        or residual_std > max(1e-5, scale * 0.03)
    ):
        raise ValueError("Oscillation fit does not support an inertial estimate")
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * (
        2 * result.cost / (len(t) - 5)
    )
    omega0 = float(np.hypot(omega, decay))
    gradient = np.array(
        [0, 0, -8 * np.pi**2 * omega / omega0**4, -8 * np.pi**2 * decay / omega0**4, 0]
    )
    period_squared = float(4 * np.pi**2 / omega0**2)
    variance = max(float(gradient @ covariance @ gradient), 0)
    return dict(
        period_squared_s2=period_squared,
        period_squared_variance=variance,
        damping_rate_s_inv=float(decay),
        cycles=cycles,
        residual_std_m=residual_std,
        measured_samples=len(t),
    )


def static_signal(record):
    if record["gravity_m_s2"] < 0.05:
        raise ValueError("Static weighing is unidentifiable in microgravity")
    rows = encoder_curve(record)
    rows = rows[rows[:, 0] >= rows[-1, 0] - 1.0]
    force = -record["spring_stiffness_n_m"] * rows[:, 1]
    if len(force) < 30 or np.ptp(force) > 0.01:
        raise ValueError("Load cell has not settled")
    return dict(
        force_n=float(force.mean()),
        force_variance=float(force.var(ddof=1) / len(force)),
        measured_samples=len(force),
    )


def estimate_mass(empty, reference, candidate):
    if [r["sample_id"] for r in (empty, reference, candidate)] != [
        None,
        "reference",
        "candidate",
    ]:
        raise ValueError("Empty, reference and candidate measurements are required")
    if len({r["method"] for r in (empty, reference, candidate)}) != 1:
        raise ValueError("Calibration and sample methods must match")
    versions = [r.get("calibration_version") for r in (empty, reference, candidate)]
    if any(versions) and (not all(versions) or len(set(versions)) != 1):
        raise ValueError("Calibration versions must match")
    if empty["method"] == "inertial":
        diagnostics = [oscillation(r) for r in (empty, reference, candidate)]
        values = [d["period_squared_s2"] for d in diagnostics]
        variances = [d["period_squared_variance"] for d in diagnostics]
    else:
        if len({r["gravity_m_s2"] for r in (empty, reference, candidate)}) != 1:
            raise ValueError("Calibration gravity must match")
        diagnostics = [static_signal(r) for r in (empty, reference, candidate)]
        values = [d["force_n"] for d in diagnostics]
        variances = [d["force_variance"] for d in diagnostics]
    v0, vr, vs = values
    denominator = vr - v0
    if denominator <= max(1e-6, 10 * np.sqrt(variances[0] + variances[1])):
        raise ValueError("Calibration signal is unidentifiable")
    mref = reference["reference_mass_kg"]
    if not isinstance(mref, (float, int)) or not np.isfinite(mref) or mref <= 0:
        raise ValueError("A positive public reference mass is required")
    mass = float(mref * (vs - v0) / denominator)
    gradient = mref * np.array(
        [(vs - vr) / denominator**2, -(vs - v0) / denominator**2, 1 / denominator]
    )
    uncertainty = float(np.sqrt(np.sum(gradient**2 * np.asarray(variances))))
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Measurements do not support a positive mass")
    return dict(
        total_mass_kg=mass,
        statistical_std_kg=uncertainty,
        uncertainty_scope="encoder_noise_and_fit_only_not_fixture_model_systematics",
        measurement_ids=[r["measurement_id"] for r in (empty, reference, candidate)],
        method=empty["method"],
        diagnostics=diagnostics,
    )
