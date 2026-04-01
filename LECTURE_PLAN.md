# Kalman Filter Intro Lecture Plan

## Purpose

This document defines a didactic concept and notebook-first structure for a short university lecture on the basic concepts of Kalman filtering.

The lecture is aimed at a robotics team in RoboCup soccer, but it intentionally avoids a math-heavy introduction. The focus is on visual intuition, uncertainty, and practical tuning insight.

## Target Format

- Medium: interactive notebook such as `marimo` or `Jupyter`
- Duration: `20-30 minutes` excluding discussion
- Style: very visual, intuition-first, minimal formulas
- Main teaching mode: plots, animations, and parameter exploration

## Audience Assumptions

- Basic technical background
- Little prior exposure to Kalman filters
- Only limited comfort with probability, matrices, or state-space notation
- Stronger motivation from robotics applications than from mathematical derivations

## Core Teaching Goals

By the end of the lecture, students should:

1. understand why filtering is needed in robotics
2. understand the Kalman filter as a repeated cycle of `prediction` and `correction`
3. interpret a Gaussian belief as a compact description of `best estimate + uncertainty`
4. develop an intuition for the meaning of uncertainty and covariance in the simple 1D case
5. understand the practical role of the three key uncertainties:
   - `P`: current estimate uncertainty
   - `Q`: process or model uncertainty
   - `R`: measurement uncertainty
6. develop a tuning intuition for what it means to trust the model more or trust the measurements more
7. recognize that the real robotics version is usually matrix-based, even if the lecture starts with a scalar example
8. leave with a qualitative understanding of where EKF and UKF fit as next steps

## High-Level Didactic Strategy

The lecture should use two layers:

1. `Motivation layer`: RoboCup ball tracking as the opening example
2. `Teaching layer`: 1D position tracking as the main explanatory vehicle

This gives a visually meaningful robotics story without forcing multivariate probability too early.

The lecture should start from a concrete problem, not from equations. Formulas should only appear after the students already understand the mechanics visually.

## Central Narrative

The lecture should revolve around one sentence:

> A Kalman filter repeatedly predicts what should happen next, then corrects that prediction with a noisy measurement, while explicitly tracking how much it trusts each source.

## Recommended Lecture Flow

1. Hook: noisy RoboCup ball tracking
2. Hidden state vs noisy observations
3. Belief as a Gaussian in 1D
4. Prediction step
5. Measurement update step
6. Tuning intuition with `P`, `Q`, and `R`
7. Full filter loop over time
8. Minimal formulas
9. Brief matrix and EKF/UKF outlook

## Detailed Section Plan

### 1. Hook: Why Filtering Exists

**Time:** `2-3 min`

**Goal:** Create the need before naming the method.

**Visual:**

- a smooth true ball trajectory
- noisy camera detections around the true trajectory
- possibly one missed or bad detection

**Key message:**

- the world has a true state
- we do not observe it directly
- sensors are noisy
- robotics still needs a useful estimate at every moment

**Suggested narration:**

"If our robot wants to intercept the ball, raw detections are not enough. We need an estimate of where the ball actually is, not just what the camera happened to report this frame."

### 2. Hidden State and Belief

**Time:** `2-3 min`

**Goal:** Introduce the idea that the filter estimates a hidden state, not just the current measurement.

**Transition:**

Move from 2D ball tracking to a simpler 1D teaching problem: tracking only the `x` position of an object.

**Visual:**

- a horizontal line representing position
- one Gaussian curve over position
- a marked mean and visible spread

**Key message:**

- the filter does not store only one number
- it stores a belief about the state
- in this lecture, belief is represented by a Gaussian

**Interpretation:**

- mean = best guess
- width = uncertainty

### 3. Prediction Step

**Time:** `3-4 min`

**Goal:** Build intuition for the motion model.

**Visual:**

- previous Gaussian belief
- predicted Gaussian shifted in the motion direction
- predicted Gaussian slightly wider than before

**Key message:**

- we use a simple motion assumption to predict the next state
- prediction moves the estimate forward
- prediction usually increases uncertainty

**Suggested narration:**

"Even without a new measurement, we can still make an educated guess about where the object should be next. But every prediction step introduces more doubt."

### 4. Measurement Update Step

**Time:** `4-5 min`

**Goal:** Show how prediction and measurement are fused.

**Visual:**

- predicted belief Gaussian
- measurement represented as a second Gaussian
- posterior Gaussian after fusion

**Key message:**

- prediction says one thing
- measurement says another thing
- the filter combines them
- the result lies in between
- the more certain source has more influence

**Important teaching move:**

Repeat the same visual with two contrast cases:

1. very precise measurement, uncertain prediction
2. noisy measurement, confident prediction

This creates intuition for trust weighting before introducing any formula.

### 5. Tuning Intuition: `P`, `Q`, and `R`

**Time:** `5-6 min`

**Goal:** Give students a strong practical feeling for covariance tuning.

This is one of the most important sections of the lecture.

**Conceptual framing:**

- `P`: how unsure am I right now?
- `Q`: how much new uncertainty does my model add each step?
- `R`: how noisy or unreliable is my sensor?

**Visuals:**

- side-by-side or slider-controlled comparisons of filter behavior
- top plot: true trajectory, measurements, estimate over time
- bottom plot: current Gaussian belief for the selected time step

**Parameter experiments:**

1. `Low R`, fixed `Q`
   - measurements are trusted strongly
   - estimate reacts quickly
   - estimate can become jittery

2. `High R`, fixed `Q`
   - measurements are trusted less
   - estimate becomes smoother
   - estimate reacts more slowly

3. `Low Q`, fixed `R`
   - model is trusted strongly
   - prediction stays relatively tight
   - filter resists sudden measurement changes

4. `High Q`, fixed `R`
   - model is trusted less
   - prediction uncertainty grows faster
   - measurements pull the estimate more strongly

**Key message:**

Tuning `Q` and `R` is not magic. It directly changes how much the filter trusts the model versus the sensor.

### 6. Full Filter Loop

**Time:** `4-5 min`

**Goal:** Show the full recursive process over time.

**Visual:**

- animation or step-through widget over multiple timesteps
- top plot: true state, noisy measurements, filtered estimate
- bottom plot: current probability belief
- optional annotation: `predict -> update -> predict -> update`

**Key message:**

- the filter is a loop, not a one-time calculation
- uncertainty grows during prediction and shrinks during correction
- filtered estimates are smoother than measurements but more responsive than pure prediction alone

### 7. Minimal Formula View

**Time:** `3-4 min`

**Goal:** Connect the visual intuition to the formal algorithm without deriving it.

**Approach:**

Show only the conceptual scalar equations and describe them in words.

Recommended content:

- prediction of state
- prediction of uncertainty
- update using measurement
- update of uncertainty

Do not derive the Kalman gain from first principles.

Instead, describe it as:

> the quantity that decides how strongly the estimate should move toward the new measurement

**Key message:**

The equations are a compact way to do exactly what the visual story already showed.

### 8. Matrix View and Robotics Outlook

**Time:** `2-3 min`

**Goal:** Connect the scalar intuition to real robotics systems.

**Message:**

- in the 1D example, `P`, `Q`, and `R` are scalars
- in real systems, they are often matrices because the state has several components
- diagonal entries describe uncertainty in each component
- off-diagonal entries describe how errors in components are coupled

Keep this brief.

Then connect to RoboCup examples:

- ball position and velocity tracking in 2D
- tracking other robots as moving obstacles
- self-localization with motion and landmark measurements

### 9. EKF and UKF Outlook

**Time:** `1-2 min`

**Goal:** Give students a forward pointer without expanding the scope.

**Message:**

- the standard Kalman filter assumes linear models
- many robotics problems are not fully linear
- EKF linearizes around the current estimate
- UKF propagates representative sigma points through nonlinear models

No derivation is needed here.

## Notebook Structure

The notebook should be organized into the following sections.

1. `Why raw measurements are not enough`
2. `Belief as a Gaussian`
3. `Prediction moves and spreads belief`
4. `Measurement update combines two sources`
5. `Trust knobs: P, Q, R`
6. `How tuning changes behavior`
7. `One full Kalman loop`
8. `Minimal formulas`
9. `From scalar intuition to matrix form`
10. `RoboCup applications`
11. `EKF and UKF outlook`

## Recommended Interactive Elements

If implemented in `marimo`, the most valuable interactive controls are:

1. a timestep slider for stepping through one run
2. a play animation for the full recursive process
3. a slider for `Q`
4. a slider for `R`
5. an optional slider for initial uncertainty `P`
6. a toggle for sudden motion changes or outlier measurements

## Recommended Visual Assets

The lecture should invest in the following visuals.

1. true trajectory vs noisy measurements
2. a single Gaussian belief over 1D position
3. prediction shown as shift plus widening
4. measurement update shown as fusion of two Gaussians
5. repeated fusion examples with different uncertainty settings
6. time-series comparison of true state, measurement, and filtered estimate
7. optional final comparison of raw vs filtered behavior under different `Q` and `R`

## Teaching Language Recommendations

Prefer intuitive language over formal language in the first half of the lecture.

Recommended phrasing:

- "belief" instead of "posterior distribution" at first
- "best guess" instead of "state estimate" at first mention
- "width" or "spread" before introducing variance or covariance
- "trust in the model" and "trust in the sensor" when discussing tuning

Only later connect these terms back to the formal names.

## Important Things To Avoid

1. starting with matrix notation
2. deriving equations before building visual intuition
3. introducing multiple robotics applications too early
4. spending too long on Gaussian algebra
5. going deep into EKF or UKF in this introductory lecture

## Suggested Closing Message

End the lecture with a short summary:

> A Kalman filter is not just a formula. It is a principled way to balance prediction and measurement under uncertainty.

> The core engineering question is always the same: how much should we trust what we expected, and how much should we trust what we just measured?

This closing directly reinforces the practical tuning intuition around `Q` and `R`.

## Possible Next Steps

After this plan, useful follow-up artifacts would be:

1. a notebook storyboard with one cell block per section
2. a concrete list of plots and animations to implement
3. a minimal 1D Kalman filter demo used by the notebook
4. a short appendix slide or notebook section for matrix notation
