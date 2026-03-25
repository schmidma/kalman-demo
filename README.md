# Kalman Filter Tutorial

An interactive `marimo` tutorial that explains the basic workings of a Kalman filter with a 1D robot x-position example.

The tutorial is designed for a short HULKs team walkthrough and then for experimentation with process noise, measurement noise, initialization, and outliers.

## Run it

```bash
uv sync
uv run marimo run main.py
```

You can also execute the marimo notebook as a plain Python script to do a quick runtime check:

```bash
uv run python main.py
```

If you want to edit the notebook-like app:

```bash
uv run marimo edit main.py
```

## What is inside

- motivation from RoboCup soccer
- probabilistic robotics notation: `bel(x_t)` and `\bar{bel}(x_t)`
- predict and update equations for the linear Gaussian case
- interactive plots for beliefs, trajectories, uncertainty, and innovations
- a minimal Python Kalman filter implementation
- outlook toward ball tracking and higher-dimensional state estimation
