import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    from dataclasses import dataclass
    from typing import Any

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from numpy.typing import NDArray

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
        max_process_noise_std = 1.0
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
        show_prediction: bool,
        show_measurement: bool,
        show_posterior: bool,
    ) -> go.Figure:
        min_process_noise_std = 0.0
        max_process_noise_std = 1.0
        min_measurement_std = 0.8
        max_measurement_std = 3.0
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
        if not any([show_prediction, show_measurement, show_posterior]):
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

    return (
        DT,
        HOOK_SEED,
        SIM_MEASUREMENT_STD,
        SIM_PROCESS_STD,
        SIM_SEED,
        STEPS,
        Settings,
        TEACH_STEP,
        FULL_LOOP_NONLINEAR_START,
        make_ball_story,
        matrix_text,
        mo,
        plot_ball_hook,
        didactic_prediction_var,
        plot_full_loop,
        plot_gaussian_intro,
        plot_motion_model_story,
        plot_step_detail,
        plot_update_story,
        run_filter,
        simulate_full_loop_demo,
        simulate_system,
        summarize_step,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kalman filters as visual reasoning under uncertainty

    This notebook is built as a short lecture, not as a reference sheet.

    The main story is simple:

    1. the world has a true state,
    2. sensors only give noisy evidence,
    3. the filter predicts what should happen next,
    4. then it corrects that prediction using the latest measurement,
    5. all the while it tracks **how uncertain it is**.

    We start with a RoboCup-style ball tracking picture, then switch to a 1D `x` example so the Gaussian beliefs stay easy to see.
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

    Imagine a robot trying to intercept the ball.

    - the black curve is the ball's true path,
    - the blue points are what the camera reports,
    - the dotted vertical line is the halfway line,
    - the circle marks the center circle of the field,
    - the detections are useful, but they jitter,
    - one bad point can easily appear,
    - the robot still needs one running estimate of where the ball actually is.

    This is the estimation problem that Kalman filtering solves.
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

    For an intro lecture, that is more geometry than we need.

    So from here on, we only track **one coordinate**: the ball's `x` position.

    That lets us draw the filter's belief directly as a 1D Gaussian curve.
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

    In this lecture, the filter's belief is a Gaussian.

    Read the picture like this:

    - the center is the current best guess,
    - the width is the uncertainty,
    - a narrow curve means "I am confident",
    - a wide curve means "I am unsure".

    The key shift in mindset is that the filter is not tracking just one number. It is tracking a **belief** about the hidden state.
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

    This model answers a simple question:

    - if nothing surprising happens,
    - where should the state move next?

    The cleanest visual story is in two steps:

    1. first, the motion model shifts the whole belief,
    2. then process noise widens that shifted belief.

    The slider controls how much extra uncertainty the prediction adds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    prediction_process_std = mo.ui.slider(
        start=0.00,
        stop=1.00,
        step=0.01,
        value=0.32,
        label="Process noise strength for the prediction view",
        show_value=True,
        debounce=True,
    )
    mo.hstack([prediction_process_std], align="start")
    return (prediction_process_std,)


@app.cell(hide_code=True)
def _(didactic_prediction_var, prediction_process_std, teaching_step):
    motion_prediction_var = didactic_prediction_var(
        teaching_step.prior_var,
        prediction_process_std.value,
    )
    return (motion_prediction_var,)


@app.cell(hide_code=True)
def _(
    mo,
    motion_prediction_var,
    plot_motion_model_story,
    prediction_process_std,
    teaching_step,
):
    mo.vstack(
        [
            mo.md(
                f"""
                In this view, the motion model moves the belief center from `{teaching_step.prior_mean:.2f}` to `{teaching_step.pred_mean:.2f}`.

                With process noise strength = `{prediction_process_std.value:.2f}`, the didactic prediction variance in `x` becomes `{motion_prediction_var:.2f}`.
                """
            ),
            mo.ui.plotly(
                plot_motion_model_story(teaching_step, prediction_process_std.value)
            ),
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Measurement update: combine two imperfect stories

    At one timestep, the filter has two sources of information:

    - the **prediction** from the motion model,
    - the **measurement** from the sensor.

    The update step fuses them into one new belief.

    Use the checkboxes to reveal the three curves one by one:

    1. predicted belief,
    2. measurement belief,
    3. updated belief.

    Then vary the measurement uncertainty. A smaller measurement covariance means the blue sensor belief is narrower and pulls the result more strongly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    update_measurement_std = mo.ui.slider(
        start=0.80,
        stop=3.00,
        step=0.05,
        value=1.20,
        label="Measurement noise strength for the update view",
        show_value=True,
        debounce=True,
    )
    show_prediction = mo.ui.checkbox(value=True, label="Show predicted belief")
    show_measurement = mo.ui.checkbox(value=False, label="Show measurement belief")
    show_posterior = mo.ui.checkbox(value=False, label="Show updated belief")
    mo.vstack(
        [
            mo.hstack([update_measurement_std], align="start"),
            mo.hstack(
                [show_prediction, show_measurement, show_posterior],
                wrap=True,
                align="start",
            ),
        ],
        gap=0.75,
    )
    return show_measurement, show_posterior, show_prediction, update_measurement_std


@app.cell(hide_code=True)
def _(
    mo,
    plot_update_story,
    prediction_process_std,
    show_measurement,
    show_posterior,
    show_prediction,
    teaching_step,
    update_measurement_std,
):
    mo.ui.plotly(
        plot_update_story(
            teaching_step,
            prediction_process_std.value,
            update_measurement_std.value,
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

    This is the part engineers care about when tuning a filter.

    In 1D, the three main uncertainties can be introduced very informally:

    - `P`: how unsure am I right now?
    - `Q`: how much new uncertainty does my motion model add each step?
    - `R`: how noisy or unreliable is my sensor?

    The plots below show the practical effect.

    - smaller `R` makes the estimate follow measurements more tightly,
    - larger `R` makes the estimate smoother and more skeptical,
    - smaller `Q` makes the model more self-confident,
    - larger `Q` lets measurements pull the estimate more strongly.
    """)
    return


@app.cell(hide_code=True)
def _(STEPS, TEACH_STEP, mo):
    process_std = mo.ui.slider(
        start=0.05,
        stop=1.00,
        step=0.01,
        value=0.32,
        label="Process uncertainty (std, controls Q)",
        show_value=True,
        debounce=True,
    )
    measurement_std = mo.ui.slider(
        start=0.20,
        stop=3.00,
        step=0.05,
        value=1.20,
        label="Measurement uncertainty (std, controls R)",
        show_value=True,
        debounce=True,
    )
    initial_pos_std = mo.ui.slider(
        start=0.5,
        stop=10.0,
        step=0.5,
        value=4.0,
        label="Initial position uncertainty (P0)",
        show_value=True,
        debounce=True,
    )
    initial_vel_std = mo.ui.slider(
        start=0.2,
        stop=4.0,
        step=0.1,
        value=1.2,
        label="Initial velocity uncertainty (P0)",
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
def _(
    initial_pos_std,
    initial_vel_std,
    inspect_step,
    measurement_std,
    mo,
    process_std,
):
    mo.vstack(
        [
            mo.md(
                r"""
                The dataset below stays fixed. Only the filter assumptions change.

                That is important: you are not changing what happened in the world. You are changing what the filter *believes* about model noise, sensor noise, and its own initial uncertainty.
                """
            ),
            mo.hstack(
                [
                    process_std,
                    measurement_std,
                    initial_pos_std,
                    initial_vel_std,
                    inspect_step,
                ],
                wrap=True,
                align="start",
            ),
        ],
        gap=1.0,
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
        interactive_run,
        interactive_settings,
        interactive_step,
        matrices,
    )


@app.cell(hide_code=True)
def _(interactive_settings, interactive_step, mo):
    measurement_message = (
        f"- a measurement is available at timestep `{interactive_step.index}`, so uncertainty in `x` shrinks from `{interactive_step.pred_var:.2f}` to `{interactive_step.var:.2f}`"
        if interactive_step.has_measurement
        else f"- no measurement is available at timestep `{interactive_step.index}`, so the filter keeps the prediction and uncertainty stays at `{interactive_step.var:.2f}`"
    )
    mo.md(f"""
    **Current tuning intuition**

    - `Q` is driven by process std = `{interactive_settings.process_std:.2f}`
    - `R` is driven by measurement std = `{interactive_settings.measurement_std:.2f}`
    - initial `P0` starts with position std = `{interactive_settings.initial_pos_std:.2f}` and velocity std = `{interactive_settings.initial_vel_std:.2f}`
    - at timestep `{interactive_step.index}`, prediction uncertainty in `x` is `{interactive_step.pred_var:.2f}`
    {measurement_message}
    """)
    return


@app.cell(hide_code=True)
def _(
    FULL_LOOP_NONLINEAR_START,
    full_loop_run,
    full_loop_simulation,
    inspect_step,
    mo,
    plot_full_loop,
):
    mo.vstack(
        [
            mo.md(
                r"""
                ## 7. The full loop over time

                Once the filter starts running, the same cycle repeats at every timestep:

                1. predict the next state,
                2. grow uncertainty through the model,
                3. compare prediction with measurement,
                4. correct the estimate,
                5. reduce uncertainty again.

                In this example, the true motion is deliberately chosen in two phases:

                - first a simple, piecewise-linear movement with a direction change,
                - later a more curved motion that no longer matches our simple prediction model as well.

                Measurements still arrive only at irregular timesteps, which makes the predict-only stretches and correction moments easier to see.

                The green band shows the filter's uncertainty around the estimate, so you can see the state evolution and confidence in one view.
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
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(interactive_settings, interactive_step, mo, plot_step_detail):
    mo.vstack(
        [
            mo.md(
                r"""
                ## 8. One timestep in detail

                This is the cleanest place to read the Kalman logic visually:

                - orange: what the model predicted,
                - blue dotted: what the sensor suggested,
                - green: what the filter believes after combining both.

                If the selected timestep has no measurement, the plot will show that the filter simply keeps the prediction.
                """
            ),
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
    ## 9. Minimal formulas after the intuition

    Only now do we introduce the equations, because by this point they should feel familiar.

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

    Conceptually:

    - `Q` grows uncertainty during prediction,
    - `R` controls how wide the sensor belief is,
    - `K` decides how strongly the estimate moves toward the measurement.
    """)
    return


@app.cell(hide_code=True)
def _(DT, mo):
    mo.md(f"""
    ## 10. Why the real robotics version uses matrices

    We taught the intuition mostly through one coordinate, but the filter underneath is already a matrix model.

    Here `dt = {DT:.1f}` and the state is:

    $$
    x_t = [\text{{position}}_t, \text{{velocity}}_t]^T
    $$

    In practice:

    - diagonal entries of a covariance matrix tell you the uncertainty of each state component,
    - off-diagonal entries tell you how errors in two components move together.

    That is why ball tracking, obstacle tracking, and self-localization are naturally matrix-based.
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
    ## 11. Back to RoboCup

    The same story extends directly to robotics tasks you actually care about:

    - **ball tracking**: estimate position and velocity in 2D,
    - **opponent tracking**: estimate where another robot is moving,
    - **self-localization**: combine motion and landmark observations.

    The details change, but the mental model stays the same:

    - predict,
    - measure,
    - fuse,
    - keep track of uncertainty.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Outlook: EKF and UKF

    The standard Kalman filter assumes linear models.

    Real robotics often breaks that assumption.

    - **EKF**: linearize the nonlinear model around the current estimate
    - **UKF**: propagate representative sigma points instead of linearizing directly

    For this lecture, the important takeaway is not the derivation. It is the continuity of the idea: even in nonlinear filters, we are still balancing prediction and measurement under uncertainty.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Closing thought

    A Kalman filter is not just a set of equations.

    It is a principled answer to one engineering question:

    > how much should I trust what I expected, and how much should I trust what I just measured?
    """)
    return


if __name__ == "__main__":
    app.run()
