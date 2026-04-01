import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    from dataclasses import dataclass
    from typing import Any

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from numpy.typing import NDArray
    from plotly.subplots import make_subplots

    Vector = NDArray[np.float64]
    Matrix = NDArray[np.float64]

    DT = 1.0
    STEPS = 60
    TEACH_STEP = 18
    FULL_LOOP_NONLINEAR_START = 24
    SIM_PROCESS_STD = 0.28
    SIM_MEASUREMENT_STD = 1.15
    SIM_SEED = 7
    HOOK_SEED = 3

    COLORS = {
        "true": "#111827",
        "measurement": "#3b82f6",
        "prediction": "#f59e0b",
        "posterior": "#059669",
        "band": "rgba(5, 150, 105, 0.18)",
        "field": "#0f172a",
        "field_line": "#94a3b8",
        "text": "#1f2937",
        "paper": "#fcfbf7",
        "plot": "#fffefb",
    }

    @dataclass(frozen=True)
    class Settings:
        process_std: float
        measurement_std: float
        initial_pos_mean: float
        initial_vel_mean: float
        initial_pos_std: float
        initial_vel_std: float

    @dataclass(frozen=True)
    class Simulation:
        controls: Vector
        truth: Matrix
        measurements: Vector
        measurement_available: NDArray[np.bool_]

    @dataclass(frozen=True)
    class Run:
        initial_mean: Vector
        initial_cov: Matrix
        pred_mean: Matrix
        pred_cov: NDArray[np.float64]
        mean: Matrix
        cov: NDArray[np.float64]
        gain: Matrix
        innovation: Vector

    @dataclass(frozen=True)
    class StepSummary:
        index: int
        prior_truth: float
        truth: float
        measurement: float
        has_measurement: bool
        prior_mean: float
        prior_var: float
        pred_mean: float
        pred_var: float
        mean: float
        var: float

    def theme() -> dict[str, Any]:
        return {
            "paper_bgcolor": COLORS["paper"],
            "plot_bgcolor": COLORS["plot"],
            "font": {
                "family": "Georgia, Times New Roman, serif",
                "color": COLORS["text"],
                "size": 16,
            },
            "margin": {"l": 56, "r": 18, "t": 64, "b": 96},
            "legend": {
                "orientation": "h",
                "x": 0.0,
                "xanchor": "left",
                "y": -0.2,
                "yanchor": "top",
            },
            "hovermode": "x unified",
            "showlegend": True,
        }

    def gaussian_pdf(x: Vector, mean: float, var: float) -> Vector:
        safe_var = max(float(var), 1e-9)
        return np.exp(-0.5 * (x - mean) ** 2 / safe_var) / np.sqrt(
            2.0 * np.pi * safe_var
        )

    def build_system(
        dt: float,
        process_std: float,
        measurement_std: float,
    ) -> tuple[Matrix, Matrix, Matrix, Matrix, Matrix]:
        a = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        b = np.array([[0.5 * dt**2], [dt]], dtype=np.float64)
        c = np.array([[1.0, 0.0]], dtype=np.float64)
        g = np.array([[0.5 * dt**2], [dt]], dtype=np.float64)
        q = (process_std**2) * (g @ g.T)
        r = np.array([[measurement_std**2]], dtype=np.float64)
        return a, b, c, q, r

    def make_controls(steps: int) -> Vector:
        controls = np.zeros(steps, dtype=np.float64)
        controls[8:18] = 0.22
        controls[18:28] = -0.08
        controls[28:36] = 0.05
        controls[36:46] = -0.18
        controls[46:54] = 0.10
        return controls

    def make_measurement_schedule(steps: int) -> NDArray[np.bool_]:
        available = np.zeros(steps, dtype=np.bool_)
        indices = np.array(
            [0, 1, 4, 7, 8, 12, 18, 19, 24, 26, 31, 32, 39, 45, 46, 52, 58],
            dtype=np.int64,
        )
        available[np.clip(indices, 0, steps - 1)] = True
        return available

    def simulate_system(
        *,
        steps: int,
        dt: float,
        process_std: float,
        measurement_std: float,
        seed: int,
    ) -> Simulation:
        a, b, _c, _, _ = build_system(dt, process_std, measurement_std)
        rng = np.random.default_rng(seed)

        truth = np.zeros((steps, 2), dtype=np.float64)
        measurements = np.full(steps, np.nan, dtype=np.float64)
        measurement_available = make_measurement_schedule(steps)
        controls = make_controls(steps)

        truth[0] = np.array([0.0, 0.9], dtype=np.float64)
        if measurement_available[0]:
            measurements[0] = truth[0, 0] + rng.normal(0.0, measurement_std)

        for t in range(1, steps):
            accel_noise = rng.normal(0.0, process_std)
            truth[t] = (
                a @ truth[t - 1] + b[:, 0] * controls[t - 1] + b[:, 0] * accel_noise
            )
            if measurement_available[t]:
                measurements[t] = truth[t, 0] + rng.normal(0.0, measurement_std)

        if measurement_available[26]:
            measurements[26] += 2.6
        return Simulation(
            controls=controls,
            truth=truth,
            measurements=measurements,
            measurement_available=measurement_available,
        )

    def simulate_full_loop_demo(
        *,
        steps: int,
        dt: float,
        measurement_std: float,
        seed: int,
        nonlinear_start: int,
    ) -> Simulation:
        baseline = simulate_system(
            steps=steps,
            dt=dt,
            process_std=SIM_PROCESS_STD,
            measurement_std=measurement_std,
            seed=seed,
        )
        rng = np.random.default_rng(seed + 101)

        controls = np.zeros(steps, dtype=np.float64)
        truth = np.zeros((steps, 2), dtype=np.float64)
        measurement_available = make_measurement_schedule(steps)
        measurements = np.full(steps, np.nan, dtype=np.float64)

        truth[0] = np.array([0.0, 0.9], dtype=np.float64)
        for t in range(1, nonlinear_start):
            velocity = 0.9 if t < nonlinear_start // 2 else -0.55
            truth[t, 1] = velocity
            truth[t, 0] = truth[t - 1, 0] + velocity * dt

        shift = truth[nonlinear_start - 1, 0] - baseline.truth[nonlinear_start - 1, 0]
        baseline_x = baseline.truth[nonlinear_start:, 0] + shift
        start_x = truth[nonlinear_start - 1, 0]
        start_v = truth[nonlinear_start - 1, 1]
        steps_after = np.arange(1, steps - nonlinear_start + 1, dtype=np.float64)
        linear_reference = start_x + start_v * steps_after * dt
        progress = np.linspace(0.0, 1.0, len(steps_after))
        deviation = baseline_x - linear_reference
        extra_bend = 0.8 * np.sin(2.8 * np.pi * progress) * progress**1.2
        truth[nonlinear_start:, 0] = linear_reference + 2.6 * deviation + extra_bend

        prev_x = truth[nonlinear_start - 1, 0]
        for t in range(nonlinear_start, steps):
            truth[t, 1] = (truth[t, 0] - prev_x) / dt
            prev_x = truth[t, 0]

        if measurement_available[0]:
            measurements[0] = truth[0, 0] + rng.normal(0.0, measurement_std)
        for t in range(1, steps):
            if measurement_available[t]:
                measurements[t] = truth[t, 0] + rng.normal(0.0, measurement_std)

        if measurement_available[26]:
            measurements[26] += 2.6

        controls[nonlinear_start // 2 - 1] = -1.45

        return Simulation(
            controls=controls,
            truth=truth,
            measurements=measurements,
            measurement_available=measurement_available,
        )

    def run_filter(
        *,
        measurements: Vector,
        controls: Vector,
        dt: float,
        settings: Settings,
    ) -> Run:
        a, b, c, q, r = build_system(
            dt,
            settings.process_std,
            settings.measurement_std,
        )
        steps = len(measurements)
        identity = np.eye(2, dtype=np.float64)

        mean = np.array(
            [settings.initial_pos_mean, settings.initial_vel_mean],
            dtype=np.float64,
        )
        cov = np.diag(
            [settings.initial_pos_std**2, settings.initial_vel_std**2]
        ).astype(np.float64)

        initial_mean = mean.copy()
        initial_cov = cov.copy()

        pred_mean = np.zeros((steps, 2), dtype=np.float64)
        pred_cov = np.zeros((steps, 2, 2), dtype=np.float64)
        filt_mean = np.zeros((steps, 2), dtype=np.float64)
        filt_cov = np.zeros((steps, 2, 2), dtype=np.float64)
        gain = np.zeros((steps, 2), dtype=np.float64)
        innovation = np.zeros(steps, dtype=np.float64)

        for t in range(steps):
            mean_bar = a @ mean + b[:, 0] * controls[t]
            cov_bar = a @ cov @ a.T + q

            if np.isfinite(measurements[t]):
                residual = measurements[t] - float((c @ mean_bar)[0])
                s = float((c @ cov_bar @ c.T + r)[0, 0])
                k = (cov_bar @ c.T / s)[:, 0]

                mean = mean_bar + k * residual
                cov = (identity - np.outer(k, c[0])) @ cov_bar
            else:
                residual = np.nan
                k = np.zeros(2, dtype=np.float64)
                mean = mean_bar
                cov = cov_bar

            pred_mean[t] = mean_bar
            pred_cov[t] = cov_bar
            filt_mean[t] = mean
            filt_cov[t] = cov
            gain[t] = k
            innovation[t] = residual

        return Run(
            initial_mean=initial_mean,
            initial_cov=initial_cov,
            pred_mean=pred_mean,
            pred_cov=pred_cov,
            mean=filt_mean,
            cov=filt_cov,
            gain=gain,
            innovation=innovation,
        )

    def summarize_step(*, run: Run, simulation: Simulation, index: int) -> StepSummary:
        if index == 0:
            prior_truth = float(simulation.truth[0, 0])
            prior_mean = float(run.initial_mean[0])
            prior_var = float(run.initial_cov[0, 0])
        else:
            prior_truth = float(simulation.truth[index - 1, 0])
            prior_mean = float(run.mean[index - 1, 0])
            prior_var = float(run.cov[index - 1, 0, 0])

        return StepSummary(
            index=index,
            prior_truth=prior_truth,
            truth=float(simulation.truth[index, 0]),
            measurement=float(simulation.measurements[index]),
            has_measurement=bool(simulation.measurement_available[index]),
            prior_mean=prior_mean,
            prior_var=prior_var,
            pred_mean=float(run.pred_mean[index, 0]),
            pred_var=float(run.pred_cov[index, 0, 0]),
            mean=float(run.mean[index, 0]),
            var=float(run.cov[index, 0, 0]),
        )

    def matrix_text(settings: Settings, dt: float) -> str:
        a, b, c, q, r = build_system(dt, settings.process_std, settings.measurement_std)
        p0 = np.diag([settings.initial_pos_std**2, settings.initial_vel_std**2])

        def block(name: str, value: Matrix) -> str:
            return (
                f"{name} =\n{np.array2string(value, precision=3, suppress_small=True)}"
            )

        return "\n\n".join(
            [
                block("A", a),
                block("B", b),
                block("C", c),
                block("Q", q),
                block("R", r),
                block("P0", p0),
            ]
        )

    def make_ball_story(steps: int, seed: int) -> tuple[Matrix, Matrix]:
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, 1.0, steps)
        x = -3.8 + 7.6 * t
        y = 0.9 * np.sin(1.3 * np.pi * t) + 0.35 * np.sin(3.0 * np.pi * t)
        truth = np.column_stack([x, y]).astype(np.float64)

        noise = rng.normal(0.0, [0.18, 0.14], size=truth.shape)
        detections = truth + noise
        detections[18] += np.array([0.55, -0.35], dtype=np.float64)
        return truth, detections

    def plot_ball_hook(truth: Matrix, detections: Matrix) -> go.Figure:
        field_outline_x = np.array([-4.5, 4.5, 4.5, -4.5, -4.5], dtype=np.float64)
        field_outline_y = np.array([-3.0, -3.0, 3.0, 3.0, -3.0], dtype=np.float64)
        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=field_outline_x,
                y=field_outline_y,
                mode="lines",
                name="Field boundary",
                line={"color": COLORS["field_line"], "width": 2},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0, 0.0],
                y=[-3.0, 3.0],
                mode="lines",
                name="Halfway line",
                line={"color": COLORS["field_line"], "width": 1.5, "dash": "dot"},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=0.65 * np.cos(theta),
                y=0.65 * np.sin(theta),
                mode="lines",
                name="Center circle",
                line={"color": COLORS["field_line"], "width": 1.2},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=truth[:, 0],
                y=truth[:, 1],
                mode="lines",
                name="True ball path",
                line={"color": COLORS["true"], "width": 4},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=detections[:, 0],
                y=detections[:, 1],
                mode="markers",
                name="Noisy detections",
                marker={
                    "color": COLORS["measurement"],
                    "size": 9,
                    "opacity": 0.72,
                },
            )
        )
        fig.update_layout(
            title="RoboCup hook: the ball has a true path, but the camera sees jitter",
            xaxis_title="field x-position [m]",
            yaxis_title="field y-position [m]",
            **theme(),
        )
        fig.update_xaxes(range=[-4.8, 4.8], zeroline=False, gridcolor="#e5e7eb")
        fig.update_yaxes(
            range=[-3.3, 3.3],
            zeroline=False,
            gridcolor="#e5e7eb",
            scaleanchor="x",
            scaleratio=1,
        )
        return fig

    def plot_gaussian_intro() -> go.Figure:
        x = np.linspace(-5.0, 5.0, 400)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=gaussian_pdf(x, 0.0, 0.35),
                mode="lines",
                name="Narrow belief",
                line={"color": COLORS["posterior"], "width": 4},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=gaussian_pdf(x, 0.0, 2.4),
                mode="lines",
                name="Wide belief",
                line={"color": COLORS["prediction"], "width": 4, "dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0, 0.0],
                y=[0.0, float(gaussian_pdf(np.array([0.0]), 0.0, 0.35)[0])],
                mode="lines",
                name="Same mean",
                line={"color": COLORS["true"], "width": 2, "dash": "dot"},
            )
        )
        fig.update_layout(
            title="Belief in 1D: same best guess, different uncertainty",
            xaxis_title="possible x-position",
            yaxis_title="density",
            **theme(),
        )
        return fig

    def _step_axis(step: StepSummary, measurement_var: float) -> Vector:
        spread = 4.8 * np.sqrt(
            max(step.prior_var, step.pred_var, step.var, measurement_var)
        )
        candidates = np.array(
            [step.prior_mean, step.pred_mean, step.mean, step.measurement, step.truth],
            dtype=np.float64,
        )
        finite_candidates = candidates[np.isfinite(candidates)]
        left = float(np.min(finite_candidates))
        right = float(np.max(finite_candidates))
        return np.linspace(left - spread, right + spread, 400)

    def didactic_prediction_var(prior_var: float, process_noise_std: float) -> float:
        return prior_var + process_noise_std**2

    def plot_motion_model_story(
        step: StepSummary,
        process_noise_std: float,
    ) -> go.Figure:
        max_process_noise_std = 3.0
        shifted_var = step.prior_var
        noisy_var = didactic_prediction_var(step.prior_var, process_noise_std)
        max_noisy_var = didactic_prediction_var(step.prior_var, max_process_noise_std)
        x = _step_axis(
            StepSummary(
                index=step.index,
                prior_truth=step.prior_truth,
                truth=step.truth,
                measurement=step.measurement,
                has_measurement=step.has_measurement,
                prior_mean=step.prior_mean,
                prior_var=step.prior_var,
                pred_mean=step.pred_mean,
                pred_var=max(max_noisy_var, shifted_var),
                mean=step.mean,
                var=step.var,
            ),
            0.0,
        )
        ymax = 1.05 * max(
            float(
                gaussian_pdf(
                    np.array([step.prior_mean]), step.prior_mean, step.prior_var
                )[0]
            ),
            float(
                gaussian_pdf(np.array([step.pred_mean]), step.pred_mean, shifted_var)[0]
            ),
            float(
                gaussian_pdf(np.array([step.pred_mean]), step.pred_mean, max_noisy_var)[
                    0
                ]
            ),
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=gaussian_pdf(x, step.prior_mean, step.prior_var),
                mode="lines",
                name="Current belief",
                line={"color": COLORS["measurement"], "width": 3, "dash": "dot"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=gaussian_pdf(x, step.pred_mean, shifted_var),
                mode="lines",
                name="Shifted by motion model only",
                line={"color": "#8b5cf6", "width": 3, "dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=gaussian_pdf(x, step.pred_mean, noisy_var),
                mode="lines",
                name="Prediction after adding process noise",
                line={"color": COLORS["prediction"], "width": 4},
            )
        )
        fig.update_layout(
            title=f"Prediction at timestep {step.index}: first shift, then widen",
            xaxis_title="x-position [m]",
            yaxis_title="density",
            **theme(),
        )
        fig.update_xaxes(range=[float(x[0]), float(x[-1])])
        fig.update_yaxes(range=[0.0, ymax])
        return fig

    def fuse_gaussians(
        prior_mean: float,
        prior_var: float,
        measurement_mean: float,
        measurement_var: float,
    ) -> tuple[float, float]:
        posterior_var = 1.0 / ((1.0 / prior_var) + (1.0 / measurement_var))
        posterior_mean = posterior_var * (
            (prior_mean / prior_var) + (measurement_mean / measurement_var)
        )
        return posterior_mean, posterior_var

    def plot_update_story(
        step: StepSummary,
        prediction_process_std: float,
        measurement_std: float,
        *,
        show_prior: bool,
        show_prediction: bool,
        show_measurement: bool,
        show_posterior: bool,
    ) -> go.Figure:
        min_process_noise_std = 0.0
        max_process_noise_std = 3.0
        min_measurement_std = 1.0
        max_measurement_std = 5.0
        prediction_var = didactic_prediction_var(step.prior_var, prediction_process_std)
        measurement_var = measurement_std**2
        posterior_mean, posterior_var = fuse_gaussians(
            step.pred_mean,
            prediction_var,
            step.measurement,
            measurement_var,
        )
        min_prediction_var = didactic_prediction_var(
            step.prior_var, min_process_noise_std
        )
        max_prediction_var = didactic_prediction_var(
            step.prior_var, max_process_noise_std
        )
        min_measurement_var = min_measurement_std**2
        max_measurement_var = max_measurement_std**2
        min_posterior_mean, min_posterior_var = fuse_gaussians(
            step.pred_mean,
            min_prediction_var,
            step.measurement,
            min_measurement_var,
        )
        max_posterior_mean, max_posterior_var = fuse_gaussians(
            step.pred_mean,
            max_prediction_var,
            step.measurement,
            max_measurement_var,
        )
        x = _step_axis(
            StepSummary(
                index=step.index,
                prior_truth=step.prior_truth,
                truth=step.truth,
                measurement=step.measurement,
                has_measurement=step.has_measurement,
                prior_mean=step.prior_mean,
                prior_var=step.prior_var,
                pred_mean=step.pred_mean,
                pred_var=max_prediction_var,
                mean=max_posterior_mean,
                var=max(max_posterior_var, max_measurement_var),
            ),
            max_measurement_var,
        )
        ymax = 1.05 * max(
            float(
                gaussian_pdf(
                    np.array([step.pred_mean]), step.pred_mean, min_prediction_var
                )[0]
            ),
            float(
                gaussian_pdf(
                    np.array([step.measurement]), step.measurement, min_measurement_var
                )[0]
            ),
            float(
                gaussian_pdf(
                    np.array([min_posterior_mean]),
                    min_posterior_mean,
                    min_posterior_var,
                )[0]
            ),
        )
        fig = go.Figure()
        if show_prior:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=gaussian_pdf(x, step.prior_mean, step.prior_var),
                    mode="lines",
                    name="Belief before prediction",
                    line={"color": "#8b5cf6", "width": 3, "dash": "dashdot"},
                )
            )
        if show_prediction:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=gaussian_pdf(x, step.pred_mean, prediction_var),
                    mode="lines",
                    name="Predicted belief",
                    line={"color": COLORS["prediction"], "width": 3},
                )
            )
        if show_measurement:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=gaussian_pdf(x, step.measurement, measurement_var),
                    mode="lines",
                    name="Measurement belief",
                    line={"color": COLORS["measurement"], "width": 3, "dash": "dot"},
                )
            )
        if show_posterior:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=gaussian_pdf(x, posterior_mean, posterior_var),
                    mode="lines",
                    name="Updated belief",
                    line={"color": COLORS["posterior"], "width": 4},
                )
            )
        if not any([show_prior, show_prediction, show_measurement, show_posterior]):
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text="Select at least one checkbox to show a belief.",
                showarrow=False,
                font={"size": 18, "color": COLORS["text"]},
            )
        fig.update_layout(
            title=f"Update at timestep {step.index}: combine prediction and measurement",
            xaxis_title="x-position [m]",
            yaxis_title="density",
            **theme(),
        )
        fig.update_xaxes(range=[float(x[0]), float(x[-1])])
        fig.update_yaxes(range=[0.0, ymax])
        return fig

    def plot_full_loop(
        simulation: Simulation,
        run: Run,
        selected_step: int,
        nonlinear_start: int,
    ) -> go.Figure:
        t = np.arange(len(simulation.measurements))
        mean = run.mean[:, 0]
        sigma = np.sqrt(run.cov[:, 0, 0])
        upper = mean + 2.0 * sigma
        lower = mean - 2.0 * sigma
        ymin = float(
            min(
                np.min(simulation.truth[:, 0]),
                np.nanmin(simulation.measurements),
                np.min(run.pred_mean[:, 0]),
                np.min(mean),
                np.min(lower),
            )
        )
        ymax = float(
            max(
                np.max(simulation.truth[:, 0]),
                np.nanmax(simulation.measurements),
                np.max(run.pred_mean[:, 0]),
                np.max(mean),
                np.max(upper),
            )
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=COLORS["band"],
                line={"color": "rgba(0,0,0,0)"},
                name="About 95% belief band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=simulation.truth[:, 0],
                mode="lines",
                name="True x",
                line={"color": COLORS["true"], "width": 3.2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=simulation.measurements,
                mode="markers",
                name="Measurements",
                marker={
                    "color": COLORS["measurement"],
                    "size": 7,
                    "opacity": 0.6,
                },
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=run.pred_mean[:, 0],
                mode="lines",
                name="Prediction",
                line={"color": COLORS["prediction"], "width": 2.2, "dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=mean,
                mode="lines",
                name="Filtered estimate",
                line={"color": COLORS["posterior"], "width": 3.4},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[nonlinear_start, nonlinear_start],
                y=[ymin, ymax],
                mode="lines",
                name="Nonlinear phase begins",
                line={"color": "#dc2626", "width": 2, "dash": "dot"},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[selected_step, selected_step],
                y=[ymin, ymax],
                mode="lines",
                name="Selected timestep",
                line={"color": "#6b7280", "width": 2, "dash": "dash"},
                hoverinfo="skip",
            )
        )
        fig.update_layout(
            title="The full loop over time: state estimate and uncertainty together",
            xaxis_title="timestep",
            yaxis_title="x-position [m]",
            **theme(),
        )
        fig.update_xaxes(gridcolor="#e5e7eb")
        fig.update_yaxes(gridcolor="#e5e7eb")
        return fig

    def plot_step_detail(step: StepSummary, measurement_std: float) -> go.Figure:
        measurement_var = measurement_std**2
        x = _step_axis(step, measurement_var)
        ymax_candidates = [
            float(
                gaussian_pdf(
                    np.array([step.prior_mean]), step.prior_mean, step.prior_var
                )[0]
            ),
            float(
                gaussian_pdf(np.array([step.pred_mean]), step.pred_mean, step.pred_var)[
                    0
                ]
            ),
            float(gaussian_pdf(np.array([step.mean]), step.mean, step.var)[0]),
        ]
        if step.has_measurement:
            ymax_candidates.append(
                float(
                    gaussian_pdf(
                        np.array([step.measurement]), step.measurement, measurement_var
                    )[0]
                )
            )
        ymax = max(ymax_candidates)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=gaussian_pdf(x, step.prior_mean, step.prior_var),
                mode="lines",
                name="Belief before prediction",
                line={"color": "#8b5cf6", "width": 3, "dash": "dashdot"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=gaussian_pdf(x, step.pred_mean, step.pred_var),
                mode="lines",
                name="Predicted belief",
                line={"color": COLORS["prediction"], "width": 3},
            )
        )
        if step.has_measurement:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=gaussian_pdf(x, step.measurement, measurement_var),
                    mode="lines",
                    name="Measurement belief",
                    line={"color": COLORS["measurement"], "width": 3, "dash": "dot"},
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=gaussian_pdf(x, step.mean, step.var),
                mode="lines",
                name="Updated belief",
                line={"color": COLORS["posterior"], "width": 4},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[step.truth, step.truth],
                y=[0.0, ymax],
                mode="lines",
                name="True x at this timestep",
                line={"color": COLORS["true"], "width": 2, "dash": "dash"},
            )
        )
        if not step.has_measurement:
            fig.add_annotation(
                x=0.5,
                y=0.92,
                xref="paper",
                yref="paper",
                text="No measurement arrived here, so the update step was skipped.",
                showarrow=False,
                font={"size": 16, "color": COLORS["text"]},
            )
        fig.update_layout(
            title=f"One selected timestep in detail: t = {step.index}",
            xaxis_title="x-position [m]",
            yaxis_title="density",
            **theme(),
        )
        return fig

    def plot_ekf_outlook() -> go.Figure:
        ekf_x = np.linspace(-2.2, 2.2, 300)
        ekf_curve = 0.35 * ekf_x**3 + 0.6 * ekf_x
        ekf_estimate = 0.8
        ekf_estimate_y = 0.35 * ekf_estimate**3 + 0.6 * ekf_estimate
        ekf_slope = 1.05 * ekf_estimate**2 + 0.6
        ekf_tangent = ekf_estimate_y + ekf_slope * (ekf_x - ekf_estimate)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ekf_x,
                y=ekf_curve,
                mode="lines",
                name="Nonlinear model",
                line={"color": COLORS["true"], "width": 3},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ekf_x,
                y=ekf_tangent,
                mode="lines",
                name="Local tangent",
                line={"color": COLORS["prediction"], "width": 3, "dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[ekf_estimate],
                y=[ekf_estimate_y],
                mode="markers",
                name="Current estimate",
                marker={"color": COLORS["posterior"], "size": 11},
            )
        )
        fig.update_layout(
            title="EKF intuition: approximate the nonlinear model locally",
            xaxis_title="state",
            yaxis_title="measurement space",
            **theme(),
        )
        fig.update_xaxes(gridcolor="#e5e7eb")
        fig.update_yaxes(gridcolor="#e5e7eb")
        return fig

    def plot_ukf_outlook() -> go.Figure:
        def ellipse_points(mean: Vector, cov: Matrix, scale: float = 2.0) -> Matrix:
            angles = np.linspace(0.0, 2.0 * np.pi, 200)
            circle = np.vstack([np.cos(angles), np.sin(angles)])
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            transform = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-9)))
            return (mean[:, None] + scale * transform @ circle).T

        state_mean = np.array([0.2, -0.25], dtype=np.float64)
        state_cov = np.array([[0.42, 0.18], [0.18, 0.24]], dtype=np.float64)

        n = 2
        alpha = 0.85
        beta = 2.0
        kappa = 0.0
        lambda_ = alpha**2 * (n + kappa) - n
        scaling = n + lambda_
        eigenvalues, eigenvectors = np.linalg.eigh(state_cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        scaled_axes = eigenvectors @ np.diag(
            np.sqrt(scaling * np.maximum(eigenvalues, 1e-9))
        )

        sigma_points = [state_mean]
        for i in range(n):
            sigma_points.append(state_mean + scaled_axes[:, i])
            sigma_points.append(state_mean - scaled_axes[:, i])
        sigma_points = np.array(sigma_points)

        weight_mean = np.full(2 * n + 1, 1.0 / (2.0 * scaling), dtype=np.float64)
        weight_cov = weight_mean.copy()
        weight_mean[0] = lambda_ / scaling
        weight_cov[0] = lambda_ / scaling + (1.0 - alpha**2 + beta)

        def nonlinear_motion(point: Vector) -> Vector:
            x, y = point
            return np.array(
                [x + 0.55 * np.sin(1.4 * y) + 0.28, y + 0.22 * x**2 - 0.12],
                dtype=np.float64,
            )

        propagated = np.array([nonlinear_motion(point) for point in sigma_points])
        propagated_mean = np.sum(weight_mean[:, None] * propagated, axis=0)
        propagated_cov = np.zeros((2, 2), dtype=np.float64)
        for weight, point in zip(weight_cov, propagated):
            delta = point - propagated_mean
            propagated_cov += weight * np.outer(delta, delta)

        prior_ellipse = ellipse_points(state_mean, state_cov)
        posterior_ellipse = ellipse_points(propagated_mean, propagated_cov)
        sigma_labels = ["0", "+x", "-x", "+y", "-y"]
        prior_axis_1 = np.column_stack(
            [state_mean - scaled_axes[:, 0], state_mean + scaled_axes[:, 0]]
        )
        prior_axis_2 = np.column_stack(
            [state_mean - scaled_axes[:, 1], state_mean + scaled_axes[:, 1]]
        )

        fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.10,
            subplot_titles=(
                "UKF prior: Gaussian and sigma points",
                "UKF after propagation: rebuilt Gaussian",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=prior_ellipse[:, 0],
                y=prior_ellipse[:, 1],
                mode="lines",
                name="Prior Gaussian",
                line={"color": COLORS["measurement"], "width": 3, "dash": "dash"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=prior_axis_1[0],
                y=prior_axis_1[1],
                mode="lines",
                name="Ellipse axis",
                line={"color": "#9ca3af", "width": 1.5, "dash": "dot"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=prior_axis_2[0],
                y=prior_axis_2[1],
                mode="lines",
                name="Ellipse axis",
                line={"color": "#9ca3af", "width": 1.5, "dash": "dot"},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sigma_points[:, 0],
                y=sigma_points[:, 1],
                mode="markers+text",
                name="Sigma points",
                text=sigma_labels,
                textposition="top center",
                marker={"color": COLORS["prediction"], "size": 10, "symbol": "diamond"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[state_mean[0]],
                y=[state_mean[1]],
                mode="markers",
                name="Prior mean",
                marker={"color": COLORS["measurement"], "size": 10},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=posterior_ellipse[:, 0],
                y=posterior_ellipse[:, 1],
                mode="lines",
                name="Re-estimated Gaussian",
                line={"color": COLORS["posterior"], "width": 3},
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=propagated[:, 0],
                y=propagated[:, 1],
                mode="markers+text",
                name="Propagated sigma points",
                text=sigma_labels,
                textposition="top center",
                marker={"color": COLORS["posterior"], "size": 10},
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[propagated_mean[0]],
                y=[propagated_mean[1]],
                mode="markers",
                name="Re-estimated mean",
                marker={"color": COLORS["posterior"], "size": 10},
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title="UKF intuition: propagate sigma points, then rebuild a Gaussian",
            **theme(),
        )
        fig.update_xaxes(title_text="x-position", gridcolor="#e5e7eb", row=1, col=1)
        fig.update_xaxes(title_text="x-position", gridcolor="#e5e7eb", row=1, col=2)
        fig.update_yaxes(title_text="y-position", gridcolor="#e5e7eb", row=1, col=1)
        fig.update_yaxes(title_text="y-position", gridcolor="#e5e7eb", row=1, col=2)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
        return fig

    return (
        DT,
        FULL_LOOP_NONLINEAR_START,
        HOOK_SEED,
        SIM_MEASUREMENT_STD,
        SIM_PROCESS_STD,
        SIM_SEED,
        STEPS,
        Settings,
        TEACH_STEP,
        make_ball_story,
        matrix_text,
        mo,
        plot_ball_hook,
        plot_ekf_outlook,
        plot_full_loop,
        plot_gaussian_intro,
        plot_motion_model_story,
        plot_step_detail,
        plot_ukf_outlook,
        plot_update_story,
        run_filter,
        simulate_full_loop_demo,
        simulate_system,
        summarize_step,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <style>
    .markdown.prose > * {
      margin-top: 0.0rem !important;
      margin-bottom: 0.0rem !important;
    }

    .markdown.prose > * + * {
      margin-top: 0.4rem !important;
    }

    .markdown.prose .paragraph {
      display: block !important;
      margin-top: 0.0rem !important;
      margin-bottom: 0.0rem !important;
    }

    .markdown.prose h1,
    .markdown.prose h2,
    .markdown.prose h3,
    .markdown.prose h4 {
      margin-top: 0.0rem !important;
      margin-bottom: 0.35rem !important;
    }

    .markdown.prose ul,
    .markdown.prose ol {
      margin-top: 0.0rem !important;
      margin-bottom: 0.0rem !important;
      padding-top: 0.0rem !important;
      padding-bottom: 0.0rem !important;
    }

    .markdown.prose li {
      margin-top: 0.0rem !important;
      margin-bottom: 0.0rem !important;
    }
    </style>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kalman Filters: A Brief Visual Introduction

    This article is an intuition-first introduction to Kalman filters.

    It assumes only a basic technical background. The goal is not a full derivation, but a clear feel for what a Kalman filter is doing and why its uncertainty terms matter.

    The core story is simple:

    1. the world has a true state,
    2. sensors only give noisy evidence,
    3. the filter predicts what should happen next,
    4. then it corrects that prediction using the latest measurement,
    5. all the while it tracks **how uncertain it is**.

    We start with a RoboCup-style ball-tracking picture, then switch to a 1D `x` example. That simplification keeps the geometry easy to see without changing the core ideas.
    """)
    return


@app.cell(hide_code=True)
def _(HOOK_SEED, make_ball_story):
    ball_truth, ball_detections = make_ball_story(steps=34, seed=HOOK_SEED)
    return ball_detections, ball_truth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Hook: why filtering exists at all

    Raw detections are not enough for action.

    A robot that reacts directly to jittery measurements will move erratically, hesitate at the wrong time, and make poor interception decisions. Even with noisy observations, it still needs one stable running estimate of where the ball actually is.
    """)
    return


@app.cell(hide_code=True)
def _(ball_detections, ball_truth, mo, plot_ball_hook):
    mo.ui.plotly(plot_ball_hook(ball_truth, ball_detections))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. From ball tracking to one coordinate

    A full ball filter is usually at least 2D, often with velocity and maybe even bounce or friction effects.

    For an introduction, that is more geometry than we need.

    So from here on, we only track **one coordinate**: the ball's `x` position.

    That lets us draw the filter's belief directly as a 1D Gaussian curve. Nothing essential is lost conceptually; we are only stripping away visual complexity.
    """)
    return


@app.cell(hide_code=True)
def _(
    DT,
    SIM_MEASUREMENT_STD,
    SIM_PROCESS_STD,
    SIM_SEED,
    STEPS,
    simulate_system,
):
    simulation = simulate_system(
        steps=STEPS,
        dt=DT,
        process_std=SIM_PROCESS_STD,
        measurement_std=SIM_MEASUREMENT_STD,
        seed=SIM_SEED,
    )
    return (simulation,)


@app.cell(hide_code=True)
def _(Settings):
    teaching_settings = Settings(
        process_std=0.32,
        measurement_std=1.20,
        initial_pos_mean=-2.0,
        initial_vel_mean=0.0,
        initial_pos_std=4.0,
        initial_vel_std=1.2,
    )
    return (teaching_settings,)


@app.cell(hide_code=True)
def _(
    DT,
    TEACH_STEP,
    run_filter,
    simulation,
    summarize_step,
    teaching_settings,
):
    teaching_run = run_filter(
        measurements=simulation.measurements,
        controls=simulation.controls,
        dt=DT,
        settings=teaching_settings,
    )
    teaching_step = summarize_step(
        run=teaching_run,
        simulation=simulation,
        index=TEACH_STEP,
    )
    return (teaching_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Belief as a Gaussian

    In this article, the filter's belief is a Gaussian.

    The center is the estimate, and the spread is the uncertainty. In plain language, the curve says where the filter thinks the state probably is.

    That is the key shift in perspective: the filter is tracking a **belief**, not just one number.
    """)
    return


@app.cell(hide_code=True)
def _(mo, plot_gaussian_intro):
    mo.ui.plotly(plot_gaussian_intro())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Motion model before prediction

    Before a Kalman filter can predict anything, it needs a **motion model**.

    The motion model encodes what we expect to happen next if nothing surprising occurs. In this simple example, it moves the belief forward in time.

    But even a good model is never perfect. That is why prediction has two effects: it shifts the belief, and it also makes the belief wider.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    prediction_process_std = mo.ui.slider(
        start=0.00,
        stop=3.00,
        step=0.01,
        value=1.50,
        label="$Q$: Process noise strength for the prediction view",
        show_value=True,
        debounce=True,
    )
    mo.hstack([prediction_process_std], align="start")
    return (prediction_process_std,)


@app.cell(hide_code=True)
def _(mo, plot_motion_model_story, prediction_process_std, teaching_step):
    mo.ui.plotly(plot_motion_model_story(teaching_step, prediction_process_std.value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Measurement update: combine two imperfect stories

    The update step fuses the model prediction with the sensor measurement.

    If the measurement is precise, the updated belief moves strongly toward it. If the measurement is noisy, the filter stays closer to its own prediction.

    The controls below let you reveal that logic step by step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    update_measurement_std = mo.ui.slider(
        start=1.00,
        stop=5.00,
        step=0.05,
        value=1.20,
        label="$R$: Measurement noise strength for the update view",
        show_value=True,
        debounce=True,
    )
    show_prior = mo.ui.checkbox(value=True, label="Show belief before prediction")
    show_prediction = mo.ui.checkbox(value=True, label="Show predicted belief")
    show_measurement = mo.ui.checkbox(value=False, label="Show measurement belief")
    show_posterior = mo.ui.checkbox(value=False, label="Show updated belief")
    mo.vstack(
        [
            mo.hstack([update_measurement_std], align="start"),
            mo.hstack(
                [show_prior, show_prediction, show_measurement, show_posterior],
                wrap=True,
                align="start",
            ),
        ],
        gap=0.75,
    )
    return (
        show_measurement,
        show_posterior,
        show_prediction,
        show_prior,
        update_measurement_std,
    )


@app.cell(hide_code=True)
def _(
    mo,
    plot_update_story,
    prediction_process_std,
    show_measurement,
    show_posterior,
    show_prediction,
    show_prior,
    teaching_step,
    update_measurement_std,
):
    mo.ui.plotly(
        plot_update_story(
            teaching_step,
            prediction_process_std.value,
            update_measurement_std.value,
            show_prior=show_prior.value,
            show_prediction=show_prediction.value,
            show_measurement=show_measurement.value,
            show_posterior=show_posterior.value,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The trust knobs: `P`, `Q`, and `R`

    At this point it is useful to name the uncertainty terms that have been implicit in the pictures so far.

    `P` is the filter's current uncertainty, `Q` is the uncertainty added by the model during prediction, and `R` is the uncertainty of the measurement.

    In practice, tuning a Kalman filter often means deciding how much to trust the model and how much to trust the sensor. The controls below let you see that balance directly.
    """)
    return


@app.cell(hide_code=True)
def _(STEPS, TEACH_STEP, mo):
    process_std = mo.ui.slider(
        start=0.05,
        stop=1.00,
        step=0.01,
        value=0.32,
        label="$Q$: Process uncertainty",
        show_value=True,
        debounce=True,
    )
    measurement_std = mo.ui.slider(
        start=0.20,
        stop=3.00,
        step=0.05,
        value=1.20,
        label="$R$: Measurement uncertainty",
        show_value=True,
        debounce=True,
    )
    initial_pos_std = mo.ui.slider(
        start=0.5,
        stop=10.0,
        step=0.5,
        value=4.0,
        label="$P_0$: Initial position uncertainty",
        show_value=True,
        debounce=True,
    )
    initial_vel_std = mo.ui.slider(
        start=0.2,
        stop=4.0,
        step=0.1,
        value=1.2,
        label="$P_0$: Initial velocity uncertainty",
        show_value=True,
        debounce=True,
    )
    inspect_step = mo.ui.slider(
        start=0,
        stop=STEPS - 1,
        step=1,
        value=TEACH_STEP,
        label="Inspect timestep",
        show_value=True,
        debounce=True,
    )
    return (
        initial_pos_std,
        initial_vel_std,
        inspect_step,
        measurement_std,
        process_std,
    )


@app.cell(hide_code=True)
def _(initial_pos_std, initial_vel_std, measurement_std, mo, process_std):
    mo.vstack(
        [
            mo.hstack(
                [process_std, measurement_std],
                wrap=True,
                align="start",
            ),
            mo.hstack(
                [initial_pos_std, initial_vel_std],
                wrap=True,
                align="start",
            ),
        ],
        gap=0.75,
    )
    return


@app.cell(hide_code=True)
def _(
    DT,
    FULL_LOOP_NONLINEAR_START,
    SIM_MEASUREMENT_STD,
    SIM_SEED,
    STEPS,
    Settings,
    initial_pos_std,
    initial_vel_std,
    inspect_step,
    matrix_text,
    measurement_std,
    process_std,
    run_filter,
    simulate_full_loop_demo,
    simulation,
    summarize_step,
):
    interactive_settings = Settings(
        process_std=process_std.value,
        measurement_std=measurement_std.value,
        initial_pos_mean=-2.0,
        initial_vel_mean=0.0,
        initial_pos_std=initial_pos_std.value,
        initial_vel_std=initial_vel_std.value,
    )
    interactive_run = run_filter(
        measurements=simulation.measurements,
        controls=simulation.controls,
        dt=DT,
        settings=interactive_settings,
    )
    interactive_step = summarize_step(
        run=interactive_run,
        simulation=simulation,
        index=inspect_step.value,
    )
    full_loop_simulation = simulate_full_loop_demo(
        steps=STEPS,
        dt=DT,
        measurement_std=SIM_MEASUREMENT_STD,
        seed=SIM_SEED,
        nonlinear_start=FULL_LOOP_NONLINEAR_START,
    )
    full_loop_run = run_filter(
        measurements=full_loop_simulation.measurements,
        controls=full_loop_simulation.controls,
        dt=DT,
        settings=interactive_settings,
    )
    matrices = matrix_text(interactive_settings, DT)
    return (
        full_loop_run,
        full_loop_simulation,
        interactive_settings,
        interactive_step,
        matrices,
    )


@app.cell(hide_code=True)
def _(
    FULL_LOOP_NONLINEAR_START,
    full_loop_run,
    full_loop_simulation,
    inspect_step,
    interactive_settings,
    interactive_step,
    mo,
    plot_full_loop,
    plot_step_detail,
):
    mo.vstack(
        [
            mo.md(
                r"""
                ## 7. The full loop over time

                Everything up to this point has focused on one moment in the cycle. This plot shows the same predict-correct logic repeated over many timesteps.

                In this example, the true motion is deliberately chosen in two phases:

                - first a simple, piecewise-linear movement with a direction change,
                - later a more curved motion that no longer matches our simple prediction model as well.

                Measurements arrive at irregular timesteps, so periods of growing uncertainty and moments of correction become easier to see.
                """
            ),
            mo.ui.plotly(
                plot_full_loop(
                    full_loop_simulation,
                    full_loop_run,
                    inspect_step.value,
                    FULL_LOOP_NONLINEAR_START,
                )
            ),
            mo.md(
                r"""
                Looking at the whole trajectory is useful, but the same run becomes easier to parse if we zoom into one selected timestep.

                If the selected timestep has no measurement, the filter simply keeps the prediction.
                """
            ),
            mo.hstack([inspect_step], align="start"),
            mo.ui.plotly(
                plot_step_detail(interactive_step, interactive_settings.measurement_std)
            ),
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Minimal formulas after the intuition

    Only now do we introduce the equations, after the visual intuition is already in place.

    The symbols map directly to the story above: $\mu$ is the estimate, $P$ is its uncertainty, $z$ is the measurement, and $K$ controls how strongly the estimate moves toward the measurement.

    Prediction:

    $$
    \bar{\mu}_t = A\mu_{t-1} + B u_t
    $$

    $$
    \bar{P}_t = A P_{t-1} A^T + Q
    $$

    Update:

    $$
    K_t = \bar{P}_t C^T (C\bar{P}_t C^T + R)^{-1}
    $$

    $$
    \mu_t = \bar{\mu}_t + K_t(z_t - C\bar{\mu}_t)
    $$

    $$
    P_t = (I - K_t C)\bar{P}_t
    $$

    Here `Q` grows uncertainty during prediction, `R` encodes measurement uncertainty, and `K` sets how strongly the estimate moves toward the measurement.
    """)
    return


@app.cell(hide_code=True)
def _(DT, mo):
    mo.md(rf"""
    ## 9. Why the real robotics version uses matrices

    We taught the intuition mostly through one coordinate, but the filter underneath is already a matrix model.

    Here `dt = {DT:.1f}` and the state is:

    $$
    x_t = [\text{{position}}_t, \text{{velocity}}_t]^T
    $$

    That matters because real systems usually estimate several coupled quantities at once.

    In practice:

    - diagonal entries of a covariance matrix tell you the uncertainty of each state component,
    - off-diagonal entries tell you how errors in two components move together.

    In the matrices below, `A` moves the state forward, `B` applies control input, `Q` injects model uncertainty, and `R` encodes sensor noise. That is why ball tracking, obstacle tracking, and self-localization are naturally matrix-based.
    """)
    return


@app.cell(hide_code=True)
def _(matrices, mo):
    mo.md(f"""
    ```text\n{matrices}\n```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Back to RoboCup

    The same ideas extend directly to robotics tasks you actually care about:

    - **ball tracking**: estimate position and velocity in 2D,
    - **opponent tracking**: estimate where another robot is moving,
    - **self-localization**: combine motion and landmark observations.

    The details change, but the mental model does not. In each case, the robot predicts, measures, fuses, and keeps track of uncertainty.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Outlook: EKF and UKF

    The standard Kalman filter assumes linear models.

    Real robotics often breaks that assumption. Once motion or sensing becomes noticeably curved, the linear story starts to fail.

    - **EKF**: linearize the nonlinear model around the current estimate
    - **UKF**: propagate representative sigma points instead of linearizing directly

    The figures below separate the two ideas: first EKF as a local tangent approximation, then UKF as a two-step 2D construction with sigma points first and a rebuilt Gaussian after nonlinear propagation.
    """)
    return


@app.cell(hide_code=True)
def _(mo, plot_ekf_outlook):
    mo.ui.plotly(plot_ekf_outlook())
    return


@app.cell(hide_code=True)
def _(mo, plot_ukf_outlook):
    mo.ui.plotly(plot_ukf_outlook())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Closing thought

    A Kalman filter is not just a set of equations.

    It is a principled answer to one engineering question:

    > how much should I trust what I expected, and how much should I trust what I just measured?

    Once that question feels intuitive, the rest of the framework starts to make sense.
    """)
    return


if __name__ == "__main__":
    app.run()
