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
    from plotly.subplots import make_subplots
    from numpy.typing import NDArray

    Vector = NDArray[np.float64]
    Matrix = NDArray[np.float64]

    TIME_STEP = 1.0
    SIMULATION_STEPS = 60
    SIMULATION_PROCESS_STD = 0.28
    SIMULATION_MEASUREMENT_STD = 1.15
    SIMULATION_SEED = 7

    @dataclass(frozen=True)
    class Settings:
        process_std: float
        measurement_std: float
        initial_pos0: float
        initial_vel0: float
        sigma_pos0: float
        sigma_vel0: float

    @dataclass(frozen=True)
    class Simulation:
        controls: Vector
        truth: Matrix
        measurements: Vector

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
    class Step:
        index: int
        prior_truth: float
        measurement: float
        truth: float
        prior_mean: float
        prior_var: float
        pred_mean: float
        pred_var: float
        mean: float
        var: float

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
        measurements = np.zeros(steps, dtype=np.float64)
        controls = make_controls(steps)

        truth[0] = np.array([0.0, 0.9], dtype=np.float64)
        measurements[0] = truth[0, 0] + rng.normal(0.0, measurement_std)

        for t in range(1, steps):
            accel_noise = rng.normal(0.0, process_std)
            truth[t] = (
                a @ truth[t - 1] + b[:, 0] * controls[t - 1] + b[:, 0] * accel_noise
            )
            measurements[t] = truth[t, 0] + rng.normal(0.0, measurement_std)

        return Simulation(controls=controls, truth=truth, measurements=measurements)

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

        mean = np.array(
            [settings.initial_pos0, settings.initial_vel0], dtype=np.float64
        )
        cov = np.diag([settings.sigma_pos0**2, settings.sigma_vel0**2]).astype(
            np.float64
        )
        initial_mean = mean.copy()
        initial_cov = cov.copy()
        identity = np.eye(2, dtype=np.float64)

        pred_mean = np.zeros((steps, 2), dtype=np.float64)
        pred_cov = np.zeros((steps, 2, 2), dtype=np.float64)
        filt_mean = np.zeros((steps, 2), dtype=np.float64)
        filt_cov = np.zeros((steps, 2, 2), dtype=np.float64)
        gain = np.zeros((steps, 2), dtype=np.float64)
        innovation = np.zeros(steps, dtype=np.float64)

        for t in range(steps):
            mean_bar = a @ mean + b[:, 0] * controls[t]
            cov_bar = a @ cov @ a.T + q

            residual = measurements[t] - float((c @ mean_bar)[0])
            s = float((c @ cov_bar @ c.T + r)[0, 0])
            k = (cov_bar @ c.T / s)[:, 0]

            mean = mean_bar + k * residual
            cov = (identity - np.outer(k, c[0])) @ cov_bar

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

    def summarize_step(*, run: Run, simulation: Simulation, index: int) -> Step:
        if index == 0:
            prior_truth = float(simulation.truth[0, 0])
            prior_mean = float(run.initial_mean[0])
            prior_var = float(run.initial_cov[0, 0])
        else:
            prior_truth = float(simulation.truth[index - 1, 0])
            prior_mean = float(run.mean[index - 1, 0])
            prior_var = float(run.cov[index - 1, 0, 0])

        return Step(
            index=index,
            prior_truth=prior_truth,
            measurement=float(simulation.measurements[index]),
            truth=float(simulation.truth[index, 0]),
            prior_mean=prior_mean,
            prior_var=prior_var,
            pred_mean=float(run.pred_mean[index, 0]),
            pred_var=float(run.pred_cov[index, 0, 0]),
            mean=float(run.mean[index, 0]),
            var=float(run.cov[index, 0, 0]),
        )

    def matrix_text(settings: Settings) -> str:
        a, b, c, q, r = build_system(
            TIME_STEP,
            settings.process_std,
            settings.measurement_std,
        )

        def block(name: str, value: Matrix) -> str:
            formatted = np.array2string(value, precision=3, suppress_small=True)
            return f"{name} =\n{formatted}"

        return "\n\n".join(
            [
                block("A", a),
                block("B", b),
                block("C", c),
                block("Q", q),
                block("R", r),
            ]
        )

    def theme() -> dict[str, Any]:
        return {
            "paper_bgcolor": "#fffdf8",
            "plot_bgcolor": "#fff8ee",
            "font": {
                "family": "Georgia, Times New Roman, serif",
                "color": "#1f2933",
            },
            "margin": {"l": 50, "r": 20, "t": 60, "b": 45},
            "legend": {"orientation": "h", "y": 1.02, "x": 0.0},
        }

    return (
        SIMULATION_MEASUREMENT_STD,
        SIMULATION_PROCESS_STD,
        SIMULATION_SEED,
        SIMULATION_STEPS,
        Settings,
        TIME_STEP,
        gaussian_pdf,
        go,
        make_subplots,
        matrix_text,
        mo,
        np,
        run_filter,
        simulate_system,
        summarize_step,
        theme,
    )


@app.cell(hide_code=True)
def _(SIMULATION_STEPS, mo):
    process_std = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.01,
        value=0.32,
        label="Process noise std",
        show_value=True,
        debounce=True,
    )
    measurement_std = mo.ui.slider(
        start=0.20,
        stop=3.0,
        step=0.05,
        value=1.25,
        label="Measurement noise std",
        show_value=True,
        debounce=True,
    )
    initial_pos0 = mo.ui.slider(
        start=-8.0,
        stop=8.0,
        step=0.5,
        value=-2.0,
        label="Initial position mean",
        show_value=True,
        debounce=True,
    )
    initial_vel0 = mo.ui.slider(
        start=-3.0,
        stop=3.0,
        step=0.1,
        value=0.0,
        label="Initial velocity mean",
        show_value=True,
        debounce=True,
    )
    sigma_pos0 = mo.ui.slider(
        start=0.5,
        stop=10.0,
        step=0.5,
        value=4.0,
        label="Initial position std",
        show_value=True,
        debounce=True,
    )
    sigma_vel0 = mo.ui.slider(
        start=0.2,
        stop=4.0,
        step=0.1,
        value=1.2,
        label="Initial velocity std",
        show_value=True,
        debounce=True,
    )
    inspect_step = mo.ui.slider(
        start=0,
        stop=SIMULATION_STEPS - 1,
        step=1,
        value=18,
        label="Inspect timestep",
        show_value=True,
        debounce=True,
    )
    return (
        initial_pos0,
        initial_vel0,
        inspect_step,
        measurement_std,
        process_std,
        sigma_pos0,
        sigma_vel0,
    )


@app.cell(hide_code=True)
def _(
    SIMULATION_MEASUREMENT_STD,
    SIMULATION_PROCESS_STD,
    SIMULATION_SEED,
    SIMULATION_STEPS,
    Settings,
    TIME_STEP,
    initial_pos0,
    initial_vel0,
    inspect_step,
    matrix_text,
    measurement_std,
    process_std,
    run_filter,
    sigma_pos0,
    sigma_vel0,
    simulate_system,
    summarize_step,
):
    settings = Settings(
        process_std=process_std.value,
        measurement_std=measurement_std.value,
        initial_pos0=initial_pos0.value,
        initial_vel0=initial_vel0.value,
        sigma_pos0=sigma_pos0.value,
        sigma_vel0=sigma_vel0.value,
    )
    simulation = simulate_system(
        steps=SIMULATION_STEPS,
        dt=TIME_STEP,
        process_std=SIMULATION_PROCESS_STD,
        measurement_std=SIMULATION_MEASUREMENT_STD,
        seed=SIMULATION_SEED,
    )
    run = run_filter(
        measurements=simulation.measurements,
        controls=simulation.controls,
        dt=TIME_STEP,
        settings=settings,
    )
    step = summarize_step(run=run, simulation=simulation, index=inspect_step.value)
    matrices = matrix_text(settings)
    return matrices, run, settings, simulation, step


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kalman filtering for a robot's x-position

    This notebook is meant as a **walk-through first** and an **interactive playground second**.
    We start with the RoboCup problem setting, then introduce the variables and formulas step by step, and only afterward turn the model into an interactive Kalman filter demo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Context: why a filter is needed at all

    Think of a robot moving along the x-axis of the field.

    We care about the robot's real state, but we never observe it directly:

    - motion commands and odometry suggest how the robot *should* move,
    - vision or other sensors provide noisy evidence about where it *seems* to be,
    - neither source is perfect on its own,
    - the filter keeps a running belief that combines both.

    The 1D x-position example is simpler than direct ball tracking, but the same logic scales to ball tracking, self-localization, and multi-sensor fusion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First visual: true state versus noisy observation

    Before introducing notation, it helps to look directly at the estimation problem:

    - the black line is the hidden true x-position,
    - the blue dots are the sensor measurements,
    - the measurements are informative, but noisy,
    - the filter will try to turn this noisy stream into a useful running belief.
    """)
    return


@app.cell
def _(go, mo, np, simulation, theme):
    _t = np.arange(len(simulation.measurements))
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_t,
            y=simulation.truth[:, 0],
            mode="lines",
            name="True position",
            line={"color": "#111827", "width": 3},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_t,
            y=simulation.measurements,
            mode="markers",
            name="Noisy measurements",
            marker={"color": "#60a5fa", "size": 8, "opacity": 0.7},
        )
    )
    _fig.update_layout(
        title="What we can observe before filtering",
        xaxis_title="timestep",
        yaxis_title="x-position [m]",
        **theme(),
    )
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## State, control, and measurement

    We now introduce the three basic objects in the model:

    - $x_t$: the hidden state,
    - $u_t$: the control input,
    - $z_t$: the measurement.

    In words:

    - the **state** is what is really going on,
    - the **control** is what we tell the robot to do,
    - the **measurement** is what the sensors report.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A Gaussian belief in one dimension

    In the Kalman filter, beliefs are Gaussian.

    For a 1D quantity, that means:

    - the **mean** says where the center of the belief is,
    - the **variance** says how spread out the belief is.

    Same center, different spread means: same best guess, different confidence.
    """)
    return


@app.cell
def _(gaussian_pdf, go, mo, np, theme):
    _x = np.linspace(-5.0, 5.0, 400)
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=gaussian_pdf(_x, 0.0, 0.4),
            mode="lines",
            name="Belief with small variance",
            line={"color": "#059669", "width": 4},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=gaussian_pdf(_x, 0.0, 2.5),
            mode="lines",
            name="Belief with large variance",
            line={"color": "#d97706", "width": 4, "dash": "dash"},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=[0.0, 0.0],
            y=[0.0, float(gaussian_pdf(np.array([0.0]), 0.0, 0.4)[0])],
            mode="lines",
            name="Shared mean",
            line={"color": "#111827", "width": 2, "dash": "dot"},
        )
    )
    _fig.update_layout(
        title="A Gaussian belief: same mean, different uncertainty",
        xaxis_title="possible x-position",
        yaxis_title="density",
        **theme(),
    )
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mean and covariance

    The Gaussian belief is written as

    $$
    bel(x_t) = \mathcal{N}(\mu_t, \Sigma_t)
    $$

    Here:

    - $\mu_t$ is the current best estimate of position and velocity,
    - $\Sigma_t$ describes how uncertain that estimate is,
    - the diagonal of $\Sigma_t$ contains variances,
    - the off-diagonal terms describe how position and velocity uncertainty are coupled.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Motion Model: Predicting the state

    The idea is simple:

    - if we know the current state estimate,
    - and we know the control input,
    - then we can predict where the robot should be next,
    - but that prediction becomes more uncertain because the world is noisy.

    The linear motion model is

    $$
    x_t = A x_{t-1} + B u_t + \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, Q)
    $$

    Term by term:

    - $A x_{t-1}$ propagates the old state forward,
    - $B u_t$ adds the effect of the control input,
    - $\epsilon_t$ captures model error and unmodeled disturbances,
    - $Q$ says how large that model uncertainty is.
    """)
    return


@app.cell(hide_code=True)
def _(gaussian_pdf, go, mo, np, step, theme):
    _spread = 4.5 * np.sqrt(max(step.prior_var, step.pred_var))
    _x = np.linspace(
        min(step.prior_mean, step.pred_mean, step.prior_truth, step.truth) - _spread,
        max(step.prior_mean, step.pred_mean, step.prior_truth, step.truth) + _spread,
        300,
    )
    _ymax = max(
        float(
            gaussian_pdf(np.array([step.prior_mean]), step.prior_mean, step.prior_var)[
                0
            ]
        ),
        float(
            gaussian_pdf(np.array([step.pred_mean]), step.pred_mean, step.pred_var)[0]
        ),
    )
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=gaussian_pdf(_x, step.prior_mean, step.prior_var),
            mode="lines",
            name="Current belief before prediction",
            line={"color": "#2563eb", "width": 3, "dash": "dot"},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=gaussian_pdf(_x, step.pred_mean, step.pred_var),
            mode="lines",
            name="Predicted belief after motion",
            line={"color": "#d97706", "width": 4},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=[step.prior_truth, step.prior_truth],
            y=[0.0, _ymax],
            mode="lines",
            name="True position before motion",
            line={"color": "#4b5563", "width": 2, "dash": "dash"},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=[step.truth, step.truth],
            y=[0.0, _ymax],
            mode="lines",
            name="True position after motion",
            line={"color": "#111827", "width": 2, "dash": "dashdot"},
        )
    )
    _fig.update_layout(
        title=f"Prediction step from t = {max(step.index - 1, 0)} to t = {step.index}",
        xaxis_title="robot x-position [m]",
        yaxis_title="density",
        **theme(),
    )
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The measurement model: what correction means

    Sensors give us evidence about the state, but not the state itself.

    In this demo, the measurement only tells us something about position. It does not directly tell us the velocity.

    The sensor model is

    $$
    z_t = C x_t + \delta_t, \qquad \delta_t \sim \mathcal{N}(0, R)
    $$

    Here:

    - $C x_t$ says which part of the state the sensor can observe (maps into the measurement dimension),
    - $\delta_t$ is measurement noise,
    - $R$ tells us how uncertain the sensor is.

    A precise sensor means a narrow likelihood; a noisy sensor means a wide likelihood.
    So $R$ belongs specifically to the update step.
    """)
    return


@app.cell(hide_code=True)
def _(gaussian_pdf, go, mo, np, settings, step, theme):
    _spread = 4.5 * settings.measurement_std
    _x = np.linspace(step.measurement - _spread, step.measurement + _spread, 300)
    _ymax = float(
        gaussian_pdf(
            np.array([step.measurement]),
            step.measurement,
            settings.measurement_std**2,
        )[0]
    )
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=gaussian_pdf(_x, step.measurement, settings.measurement_std**2),
            mode="lines",
            name="Measurement likelihood",
            line={"color": "#2563eb", "width": 4, "dash": "dot"},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=[step.truth, step.truth],
            y=[0.0, _ymax],
            mode="lines",
            name="True position at this timestep",
            line={"color": "#111827", "width": 2, "dash": "dash"},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=[step.measurement, step.measurement],
            y=[0.0, _ymax],
            mode="lines",
            name="Measured position at this timestep",
            line={"color": "#2563eb", "width": 2, "dash": "dot"},
        )
    )
    _fig.update_layout(
        title="What the sensor says at this timestep",
        xaxis_title="robot x-position [m]",
        yaxis_title="likelihood",
        **theme(),
    )
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Bayesian recursion before Kalman formulas

    In *Probabilistic Robotics*, we

    1. **Predict** with the motion model.
    2. **Update** with the measurement likelihood.

    The Kalman filter is the linear-Gaussian case where this recursion can be written directly in terms of means and covariances.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prediction equations in the Kalman filter

    Once we assume a linear-Gaussian model, prediction becomes:

    $$
    \bar{\mu}_t = A\mu_{t-1} + B u_t
    $$

    $$
    \bar{\Sigma}_t = A\Sigma_{t-1}A^T + Q
    $$

    In words:

    - move the mean with the motion model,
    - grow the uncertainty with the model noise $Q$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Update equations in the Kalman filter

    After receiving a measurement, the update becomes:

    $$
    K_t = \bar{\Sigma}_t C^T (C\bar{\Sigma}_t C^T + R)^{-1}
    $$

    $$
    \mu_t = \bar{\mu}_t + K_t(z_t - C\bar{\mu}_t)
    $$

    $$
    \Sigma_t = (I - K_t C)\bar{\Sigma}_t
    $$

    The roles are:

    - $z_t - C\bar{\mu}_t$ is the **innovation**,
    - $K_t$ is the **Kalman gain**,
    - the mean is corrected,
    - the covariance usually shrinks according to the sensor uncertainty $R$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The state vector and its dimensions

    In this tutorial, the hidden state is

    $$
    x_t = [position_t, velocity_t]^T
    $$
    """)
    return


@app.cell(hide_code=True)
def _(
    initial_pos0,
    initial_vel0,
    measurement_std,
    mo,
    process_std,
    sigma_pos0,
    sigma_vel0,
):
    mo.vstack(
        [
            mo.md(
                """
                ## Now make the model interactive

                From here on, you can manipulate the filter parameters directly with sliders and inspect how the belief changes.
                A good order is: first change the initial belief, then tune `Q` and `R`, then inspect one timestep, and only afterward look at the full trajectory.
                """
            ),
            mo.hstack(
                [
                    initial_pos0,
                    initial_vel0,
                    sigma_pos0,
                    sigma_vel0,
                    process_std,
                    measurement_std,
                ],
                wrap=True,
                align="start",
            ),
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(TIME_STEP, matrices, mo):
    mo.vstack(
        [
            mo.md(
                f"""
                ## What the matrices look like numerically

                For the current tuning and `dt = {TIME_STEP:.1f}`, the matrices are:

                - `A` propagates position and velocity forward,
                - `B` injects the acceleration control,
                - `C` says that we only measure position,
                - `Q` encodes process uncertainty,
                - `R` encodes measurement uncertainty.
                """
            ),
            mo.md(f"```text\n{matrices}\n```"),
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(gaussian_pdf, go, inspect_step, mo, np, settings, step, theme):
    _spread = 4.5 * np.sqrt(max(step.pred_var, step.var, settings.measurement_std**2))
    _x = np.linspace(
        min(step.pred_mean, step.mean, step.measurement, step.truth) - _spread,
        max(step.pred_mean, step.mean, step.measurement, step.truth) + _spread,
        300,
    )
    _ymax = max(
        float(
            gaussian_pdf(np.array([step.pred_mean]), step.pred_mean, step.pred_var)[0]
        ),
        float(
            gaussian_pdf(
                np.array([step.measurement]),
                step.measurement,
                settings.measurement_std**2,
            )[0]
        ),
        float(gaussian_pdf(np.array([step.mean]), step.mean, step.var)[0]),
    )

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=gaussian_pdf(_x, step.pred_mean, step.pred_var),
            mode="lines",
            name="Predicted belief before the measurement",
            line={"color": "#d97706", "width": 3},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=gaussian_pdf(_x, step.measurement, settings.measurement_std**2),
            mode="lines",
            name="Measurement likelihood from the sensor",
            line={"color": "#2563eb", "width": 3, "dash": "dot"},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=gaussian_pdf(_x, step.mean, step.var),
            mode="lines",
            name="Posterior belief after the update",
            line={"color": "#059669", "width": 4},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=[step.truth, step.truth],
            y=[0.0, _ymax],
            mode="lines",
            name="True position at this timestep",
            line={"color": "#111827", "width": 2, "dash": "dash"},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=[step.measurement, step.measurement],
            y=[0.0, _ymax],
            mode="lines",
            name="Measured position at this timestep",
            line={"color": "#2563eb", "width": 2, "dash": "dot"},
        )
    )
    _fig.update_layout(
        title=f"One step in detail at t = {step.index}",
        xaxis_title="robot x-position [m]",
        yaxis_title="density",
        **theme(),
    )
    mo.vstack(
        [
            mo.md(
                """
                ## Interactive step view

                Start with one timestep only. This is the cleanest place to see the Kalman logic.

                - orange: predicted belief $\\bar{bel}(x_t)$
                - blue dotted: measurement likelihood $p(z_t \\mid x_t)$
                - green: posterior belief $bel(x_t)$

                The controls here are only for *which* timestep you inspect. The tuning sliders above still control the actual filter behavior.
                """
            ),
            mo.hstack([inspect_step], align="start"),
            mo.ui.plotly(_fig),
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(go, mo, np, run, simulation, theme):
    _t = np.arange(len(simulation.measurements))
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_t,
            y=simulation.truth[:, 0],
            mode="lines",
            name="True position",
            line={"color": "#111827", "width": 3},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_t,
            y=simulation.measurements,
            mode="markers",
            name="Noisy measurements",
            marker={"color": "#60a5fa", "size": 8, "opacity": 0.65},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_t,
            y=run.pred_mean[:, 0],
            mode="lines",
            name="Predicted position before update",
            line={"color": "#d97706", "width": 2, "dash": "dash"},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_t,
            y=run.mean[:, 0],
            mode="lines",
            name="Filtered estimate after update",
            line={"color": "#059669", "width": 3},
        )
    )
    _fig.update_layout(
        title="Repeated predict and update over time",
        xaxis_title="timestep",
        yaxis_title="x-position [m]",
        **theme(),
    )
    mo.vstack(
        [
            mo.md(
                """
                ## Full filtering loop over time

                Now the single-step idea is repeated over the whole sequence.
                The filter alternates between prediction and correction at every timestep.
                This is where you can see whether your tuning produces a stable, useful estimate over longer horizons.
                """
            ),
            mo.ui.plotly(_fig),
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(go, mo, np, run, theme):
    _t = np.arange(len(run.mean))
    _mean = run.mean[:, 0]
    _sigma = np.sqrt(run.cov[:, 0, 0])
    _upper = _mean + 2.0 * _sigma
    _lower = _mean - 2.0 * _sigma

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=np.concatenate([_t, _t[::-1]]),
            y=np.concatenate([_upper, _lower[::-1]]),
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.18)",
            line={"color": "rgba(0,0,0,0)"},
            name="Uncertainty band (about 95%)",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_t,
            y=_mean,
            mode="lines",
            name="Filtered mean",
            line={"color": "#b45309", "width": 3},
        )
    )
    _fig.update_layout(
        title="Uncertainty shrinks and grows with the evidence",
        xaxis_title="timestep",
        yaxis_title="x-position [m]",
        **theme(),
    )
    mo.vstack(
        [
            mo.md(
                """
                ## Uncertainty is part of the state estimate

                A Kalman filter is not just estimating *where* the robot is.
                It is also estimating *how certain* it is about that estimate.
                If you change `Q`, `R`, or the initial uncertainty above, this band should react in an intuitive way.
                """
            ),
            mo.ui.plotly(_fig),
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(go, make_subplots, mo, np, run, theme):
    _t = np.arange(len(run.innovation))
    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Innovation over time",
            "Kalman gain for position over time",
        ),
    )
    _fig.add_trace(
        go.Bar(
            x=_t,
            y=run.innovation,
            name="Innovation (measurement - predicted observation)",
            marker_color="#2563eb",
            opacity=0.75,
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_t,
            y=run.gain[:, 0],
            mode="lines+markers",
            name="Kalman gain for position",
            line={"color": "#dc2626", "width": 3},
        ),
        row=2,
        col=1,
    )
    _fig.update_layout(
        title="Innovation and Kalman gain",
        **theme(),
    )
    _fig.update_yaxes(title_text="innovation [m]", row=1, col=1)
    _fig.update_yaxes(title_text="gain", range=[0, 1.05], row=2, col=1)
    _fig.update_xaxes(title_text="timestep", row=2, col=1)
    mo.vstack(
        [
            mo.md(
                """
                ## Why the estimate moves: innovation and gain

                The innovation says how surprising the current measurement is.
                The Kalman gain says how strongly that surprise should change the estimate.
                If the gain is large, measurements pull the estimate strongly. If it is small, the model dominates.
                """
            ),
            mo.ui.plotly(_fig),
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Minimal Python implementation

    After the concepts, the code should look much less mysterious:

    ```python
    mean_bar = a @ mean + b[:, 0] * controls[t]
    cov_bar = a @ cov @ a.T + q

    residual = measurements[t] - float((c @ mean_bar)[0])
    s = float((c @ cov_bar @ c.T + r)[0, 0])
    k = (cov_bar @ c.T / s)[:, 0]

    mean = mean_bar + k * residual
    cov = (identity - np.outer(k, c[0])) @ cov_bar
    ```

    This is exactly the narrative we built up:

    1. move the belief through the motion model,
    2. compare prediction and measurement,
    3. correct the estimate,
    4. update the uncertainty.
    """)
    return


@app.cell(hide_code=True)
def _(mo, simulation):
    mo.md(f"""
    ## Discussion and outlook

    The simulated robot starts at x = `{simulation.truth[0, 0]:.1f}` m with velocity `{simulation.truth[0, 1]:.1f}` m/s.

    Good discussion prompts for the team:

    - What happens if `R` is too small and the filter trusts vision too much?
    - What happens if `Q` is too small and the motion model is overconfident?
    - Which hidden variables would be added for a ball-tracking filter?
    - How would the model change for 2D position, heading, or multiple sensors?

    Outlook: the same recursion extends naturally to **ball tracking**. The state, matrices, and sensor model get richer, but the basic predict/update story stays the same.
    """)
    return


if __name__ == "__main__":
    app.run()
