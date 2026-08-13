# Flow Matching ODE Solvers, Illustrated

<div class="ln-byline">2026-08-06 · about 22 min read · Yufeng Jin</div>

<p class="ln-lead" markdown>Sampling an image = numerically integrating an ODE. This note takes five classic solvers apart and explains each one (<span class="fms-t-euler">Euler</span>, <span class="fms-t-midpoint">Midpoint</span>, <span class="fms-t-heun">Heun</span>, <span class="fms-t-rk4">RK4</span>, <span class="fms-t-dopri5">Dopri5</span>) — every animation is a **real integration**, and every data figure comes from a reproducible experiment.</p>

<figure class="fms-panel">
  <canvas id="heroCv" height="380" aria-label="Flow matching vector field animation: particles flowing from Gaussian noise to eight modes"></canvas>
  <figcaption><b>Cover figure · </b>512 particles integrated in real time along the exact closed-form solution of the 8-Gaussians flow matching velocity field: from Gaussian noise at the center out to the eight modes marked with red ×. This is not a schematic animation — these are genuine trajectories computed frame by frame in your browser.</figcaption>
</figure>

---

<div class="ln-eyebrow">Opening</div>

## 1 · Introduction

You have trained a flow matching model. When you sample, there is always that one line in the code — `x = solver.step(x, t)` — but what actually happens inside it? Why does Stable Diffusion 3 default to 28 steps rather than 1000? Why do some people use Euler, others Heun, and others the impressive-sounding DPM-Solver? How much compute does swapping the solver really save?

This note answers those questions through a **fully reproducible 2D toy experiment**. We train a real flow matching model (two 2D datasets, 15000 steps, fixed seed), then put five classic ODE solvers — <span class="fms-t-euler">Euler</span>, <span class="fms-t-midpoint">Midpoint</span>, <span class="fms-t-heun">Heun</span>, <span class="fms-t-rk4">RK4</span>, <span class="fms-t-dopri5">Dopri5</span> — head to head on the same field.

Read it however you like, but two ground rules apply: every interactive figure on this page is **real numerical integration inside your browser** (not a pre-rendered GIF), and every number in every data figure comes from an actual run of the companion code `fm_solvers.py`. By the end you should be able to answer: *given an NFE budget, which solver should I pick, and why.*

<div class="ln-howto"><b>How to read this</b>: if you are already comfortable with ODEs and
numerical integration, jump straight to Section 4. If you only want the conclusions, read the
five takeaways in Section 8.3 and the decision table in Section 9. Sections 4-6 are the main
method line; each unfolds as intuition, mechanism/math, measurement, lesson.</div>

<div class="ln-eyebrow">Opening · why it matters</div>

## 2 · Motivation: why a generative modeling engineer should understand solvers

Three reasons, each more practical than the last:

- **NFE is money.** The cost of generating one image ≈ number of network forward passes × cost per forward pass. Once the model is fixed, the cost per forward pass is fixed too — what the solver decides is *how many forward passes the same quality costs you*.
- **The gap is a factor of 3, and it is free.** In the experiments here, for one and the same trained model to reach quality close to the sampling floor: Euler needs NFE=40, RK4 only 12 (Section 8). No model changes, no retraining — just a few different lines of solver code.
- **The conclusions are not obvious.** At very low budgets a higher-order method can actually be worse; when the field is poorly trained Euler overtakes RK4; and the "automatic transmission" of adaptive methods is not optimal everywhere either. Choosing by intuition is an easy way to choose wrong — which is why every claim here comes with data.

<div class="ln-eyebrow">Prerequisites</div>

## 3 · Prerequisites

### 3.1 · What an ODE and an initial value problem are

An ordinary differential equation (ODE) looks like this:

$$
\frac{dx}{dt} = f(x, t)
$$

It states something very plain: "a point sitting at position $x$ at time $t$ has **velocity** $f(x,t)$". That $f$ is called a **vector field** (向量场) — every point in space carries a little arrow (the pale short strokes in the cover figure are exactly those). Add a starting point $x(0)=x_0$ and you have an **initial value problem** (IVP, 初值问题): set off from $x_0$, follow the arrows at every instant, and the trajectory $x(t)$ you trace out is the solution.

Only a handful of $f$ admit a closed-form solution (for instance $dx/dt=x^2$, which Section 5 uses as a test case). When $f$ is a neural network, an analytic solution is out of the question — **numerical solution is the only option**, and that is exactly the subject of this note.

### 3.2 · How a generative model turns into an ODE

Viewing "generation" as "solving an ODE" was not invented by flow matching; it is a river with several tributaries. Neural ODEs / continuous normalizing flows model generation directly as integrating a learned field<sup>[[1]](#refs)</sup>; the stochastic sampling process of score-based diffusion models<sup>[[3]](#refs)</sup> was shown to admit a deterministic "probability flow ODE" with exactly the same marginals<sup>[[2]](#refs)</sup>; and flow matching<sup>[[4]](#refs)</sup> / rectified flow<sup>[[5]](#refs)</sup> skip the SDE altogether, training a velocity field with a plain regression objective. It is this last variety (the linear/OT path) that we use here:

$$
x_t = (1-t)\,x_0 + t\,x_1, \qquad \text{regression target}\ \ v = x_1 - x_0
$$

At training time we draw random points on the segment joining noise $x_0$ and data $x_1$, and let the network $v_\theta(x,t)$ regress "which way things flow on average". At sampling time we run it in reverse, starting from $x_0 \sim \mathcal{N}(0, I)$ and solving the initial value problem:

$$
\frac{dx}{dt} = v_\theta(x, t), \qquad t: 0 \to 1
$$

Wherever the particle lands at $t=1$ is the sample. Hence:

**Generation quality = how accurately the field was learned (model error) × how accurately it is integrated (discretization error).** The former is the training's business, the latter the solver's — and this note is only about the latter: same field, different solvers, how far apart they end up.

### 3.3 · Getting started with numerical integration: discretization and "order"

Numerical solution boils down to one sentence: chop $t\in[0,1]$ into $N$ small steps, and use a **finite number** of field evaluations per step to approximate the true trajectory. Two key notions:

- **Local truncation error** (局部截断误差): the mistake made in a single step. Compare the method's update formula term by term against the Taylor expansion of the true solution; if they agree up to the $dt^p$ term, the error of that step is $O(dt^{p+1})$.
- **Global error** (全局误差): the mistake accumulated over the whole trip. There are $N=1/dt$ steps in total, and the errors roughly add up: $O(dt^{p+1})\cdot(1/dt) = O(dt^p)$. Such a method is called a **$p$-th order method**.

"Order" (阶) is the key that unlocks everything: halve the step size of a $p$-th order method and the error drops by a factor of $2^p$, which on a log-log plot is a straight line of slope $-p$ — Section 5 measures this live. One boundary remark: none of the flow matching fields in this note are "stiff", so explicit methods are entirely sufficient; stiff problems call for implicit methods, and that is a whole other book<sup>[[7]](#refs)</sup>.

### 3.4 · The unit of account: NFE

When comparing solvers, the horizontal axis is always **NFE** (number of function evaluations, i.e. network forward passes) rather than step count: one RK4 step costs 4 forward passes, one Euler step only 1. "10 RK4 steps vs 10 Euler steps" is unfair; "RK4 at NFE=40 vs Euler at NFE=40" is fair — *NFE is money, and it is the horizontal axis of every figure in this note.*

<div class="ln-eyebrow">Method 01 · fixed step</div>

## 4 · Fixed-step solvers: how a single step is taken

With the prerequisites in place, on to the main event. Every "one-step method" answers the same question: *given the velocity at the current point, take a step forward of size dt — where do we land?* They differ only in how many times they "peek" at the field, and how they combine those peeks — peek more cleverly, get a higher order.

Rather than staring at formulas, let us take a step apart by hand. The field in Figure 1 is a uniformly rotating field (the true solution is a circular arc, so the curvature is obvious at a glance): switch solvers, drag dt, and watch how each method assembles the step out of 1–4 "peeks", and how the single-step error grows.

<figure class="fms-panel">
  <div class="fms-controls">
    <span class="fms-seg" id="anatomySeg" role="group" aria-label="Select solver"></span>
    <label>Step size dt <input type="range" id="anatomyDt" min="0.15" max="1.35" step="0.05" value="0.9">
      <span class="fms-readout" id="anatomyDtVal"></span></label>
  </div>
  <canvas id="anatomyCv" height="360" aria-label="Anatomy of a single step: the stage velocity evaluations compared with the true arc"></canvas>
  <div class="fms-stats">
    <span><span class="k">NFE this step</span> <b id="anatomyNfe"></b></span>
    <span><span class="k">Single-step error</span> <b id="anatomyErr"></b></span>
    <span class="k" id="anatomyDesc"></span>
  </div>
  <figcaption><b>Figure 1 · Anatomy of a single step (interactive).</b> One step on a uniformly rotating field: the thick gray line is the true arc, the colored arrow is the direction the solver actually takes, the dashed gray arrows and small dots are the intermediate "peeks" (the stage evaluations k₁…k₄), and the dashed red segment is the single-step error. Switch solvers to compare: at the same dt, the more peeks and the smarter the combination, the closer the landing point is to the true solution.</figcaption>
</figure>

### NFE bookkeeping

| Solver | Order | Evaluations per step | What an evaluation buys | NFE at steps=10 |
|---|---|---|---|---|
| <span class="fms-chip fms-c-euler"></span>Euler | 1 | 1 | Take the full step straight along the starting velocity k₁ | 10 |
| <span class="fms-chip fms-c-midpoint"></span>Midpoint | 2 | 2 | Scout half a step ahead, then take the full step with the **midpoint velocity** k₂ | 20 |
| <span class="fms-chip fms-c-heun"></span>Heun | 2 | 2 | Scout a full step ahead, then correct with the **average of the endpoints** (k₁+k₂)/2 | 20 |
| <span class="fms-chip fms-c-rk4"></span>RK4 | 4 | 4 | Two midpoints + one endpoint, weighted as (k₁+2k₂+2k₃+k₄)/6 | 40 |

<div class="ln-eyebrow">Method 02 · order of convergence</div>

## 5 · Order of convergence: why higher order "saves money"

Section 3.3 said it: halve the step size of a $p$-th order method and the error drops by $2^p$, a line of slope $-p$ on a log-log plot. This is not just theory — the convergence-order unit test in the companion code `test_solvers.py` asserts exactly these ratios (Euler 2×, Heun/Midpoint 4×, RK4 16×), and Figure 2 lets you see it with your own eyes.

Figure 2 solves $dx/dt = x^2$, $x(0)=\tfrac12$ (true solution $x(1)=1$) live in your browser, plotting the error NFE by NFE. Note that the horizontal axis has already been converted to NFE: each RK4 step costs 4× more, but a slope of −4 earns that back very quickly.

!!! note "An amusing trap"
    Why not use the more convenient $dx/dt = x$? Because on a **linear** ODE the single-step updates of Midpoint and Heun are algebraically identical (both are exactly multiplication by $1+h+h^2/2$), so the two curves would coincide digit for digit and look like a plotting bug — to see "same order, different error constant" you need a nonlinear equation. We really did step on this rake.

<figure class="fms-panel">
  <div class="fms-legend" id="convLegend"></div>
  <canvas id="convCv" height="380" aria-label="Order-of-convergence plot: log-log curves of error vs NFE for four solvers on an analytic ODE"></canvas>
  <figcaption><b>Figure 2 · Order of convergence, measured (computed live in the browser; interactive: hover for readouts).</b> Endpoint error vs NFE (log-log) for four solvers on dx/dt=x². The dashed gray lines are slope −1/−2/−4 references: Euler hugs −1, Midpoint and Heun hug −2 (two parallel lines with different error constants), RK4 hugs −4. At the same NFE=40, the four curves can differ by as much as 6 orders of magnitude.</figcaption>
</figure>

<div class="ln-lesson"><b>Lesson</b>: order is not a paper specification — at the same NFE=40 the
four curves differ in endpoint error by up to six orders of magnitude. Same-order Midpoint and
Heun are two parallel lines: the order sets the slope, the error constant sets the intercept.
Verifying convergence order requires a <b>nonlinear</b> equation: on a linear ODE the one-step
updates of Midpoint and Heun are algebraically identical, so the two curves coincide digit for
digit and the plot looks broken.</div>

<div class="ln-eyebrow">Method 03 · adaptive step</div>

## 6 · Adaptive step size: how Dopri5 paces itself

Fixed step sizes have an awkward property: where the field is gentle, small steps waste effort; where it bends sharply, large steps crash. Dormand–Prince 5(4)<sup>[[6]](#refs)</sup> — the machinery behind SciPy's `RK45`, MATLAB's `ode45`, and torchdiffeq's default solver — solves it by **computing two answers at every step**: 7 stage evaluations assemble a 5th-order solution $y_5$ and a 4th-order solution $y_4$, and their difference is a free error estimate:

$$
\mathrm{err} = \mathrm{rms}\!\left(\frac{y_5 - y_4}{\mathrm{atol} + \mathrm{rtol}\cdot\max(|x|, |y_5|)}\right)
$$

- $\mathrm{err} \le 1$: accept the step, and reuse the last stage derivative in the next step (FSAL, so 7 stages cost only 6 new evaluations);
- $\mathrm{err} > 1$: **reject**, shrink dt and retry (the evaluations of that attempt are wasted);
- either way, update the step size by `dt ×= clip(0.9·err^(−1/5), 0.2, 5)` (the classic step-size controller: safety factor 0.9, clipping to prevent oscillation<sup>[[7]](#refs)</sup>).

Figure 3 runs it once on the real 8-Gaussians flow field (512 particles sharing one step size, matching the implementation in `fm_solvers.py`). This field is nearly straight early on, and near $t=1$ the particles have to squeeze into narrow peaks of σ=0.08 — watch how dt automatically goes "loose early, tight late".

<figure class="fms-panel">
  <div class="fms-controls">
    <label>rtol
      <span class="fms-seg" id="dpSeg" role="group" aria-label="Select tolerance"></span>
    </label>
  </div>
  <canvas id="dpCv" height="300" aria-label="Dopri5 adaptive step size bar chart: dt of each step, accepted and rejected"></canvas>
  <div class="fms-stats">
    <span><span class="k">Accepted</span> <b id="dpAcc"></b></span>
    <span><span class="k">Rejected</span> <b id="dpRej"></b></span>
    <span><span class="k">Total NFE</span> <b id="dpNfe"></b></span>
    <span><span class="k">Min/max dt</span> <b id="dpRange"></b></span>
  </div>
  <figcaption><b>Figure 3 · Dopri5 pacing itself (interactive: switch rtol).</b> Each bar is one attempt: the horizontal axis is the t interval it covers, the height is log(dt); solid orange = accepted, red outline = rejected and retried. Early on (where the field is nearly straight) it strides ahead; approaching t=1 (as particles squeeze into the narrow peaks) it automatically takes mincing steps and rejections appear. Tighten rtol and the steps get denser overall while NFE climbs.</figcaption>
</figure>

### Butcher tableau (Dormand–Prince 5(4))

These coefficients are the source of `_DP_A / _DP_B5 / _DP_B4` in `fm_solvers.py`. Row $i$ says that stage $i$ evaluates the field at $t + c_i\cdot dt$, at a position formed as the linear combination of the earlier stages with weights $a_{ij}$:

<table class="fms-butcher">
<tr><td class="cl">0</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td class="cl">1/5</td><td>1/5</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td class="cl">3/10</td><td>3/40</td><td>9/40</td><td></td><td></td><td></td><td></td></tr>
<tr><td class="cl">4/5</td><td>44/45</td><td>−56/15</td><td>32/9</td><td></td><td></td><td></td></tr>
<tr><td class="cl">8/9</td><td>19372/6561</td><td>−25360/2187</td><td>64448/6561</td><td>−212/729</td><td></td><td></td></tr>
<tr><td class="cl">1</td><td>9017/3168</td><td>−355/33</td><td>46732/5247</td><td>49/176</td><td>−5103/18656</td><td></td></tr>
<tr><td class="cl">1</td><td>35/384</td><td>0</td><td>500/1113</td><td>125/192</td><td>−2187/6784</td><td>11/84</td></tr>
<tr><td class="cl tp"><i>b</i><sub>5</sub></td><td class="tp">35/384</td><td class="tp">0</td><td class="tp">500/1113</td><td class="tp">125/192</td><td class="tp">−2187/6784</td><td class="tp">11/84</td></tr>
<tr><td class="cl"><i>b</i><sub>4</sub></td><td>5179/57600</td><td>0</td><td>7571/16695</td><td>393/640</td><td>−92097/339200</td><td>187/2100</td></tr>
</table>

*Notice that $b_5$ is exactly the $a$ row of the final stage — so "this step's 5th-order solution" is precisely "the next step's stage 1 evaluation point", and that is where FSAL (First Same As Last) saves an evaluation. The b₄ row also has a trailing 1/40 (the weight of stage 7).*

<div class="ln-lesson"><b>Lesson</b>: adaptivity is not "higher order" — it turns choosing the step
size from a hand-run grid sweep into setting a single <code>rtol</code>. The price is computing a
5th- and a 4th-order solution every step to get a free error estimate; FSAL makes seven stages cost
only six fresh network evaluations, while a rejected step is evaluation spent for nothing.</div>

<div class="ln-eyebrow">Hands-on · same field</div>

## 7 · Sample it yourself: one field, five ways to walk it

Now let us put Sections 4–6 together for real: 1500 particles start from $\mathcal{N}(0,I)$ and are integrated to $t=1$ on the exact 8-Gaussians flow field. Change the solver, drag the step count, and watch the scatter cloud condense from "one blurry ring" into eight sharp peaks. A recommended comparison tour: *Euler·2 steps → Euler·8 steps → Midpoint·4 steps (NFE=8) → RK4·2 steps (NFE=8)* — same NFE, very different shapes, and that difference is what "how you spend the money" means.

<figure class="fms-panel">
  <div class="fms-controls">
    <span class="fms-seg" id="pgSeg" role="group" aria-label="Select solver"></span>
    <label id="pgStepsLab">Steps <input type="range" id="pgSteps" min="0" max="8" step="1" value="3">
      <span class="fms-readout" id="pgStepsVal"></span></label>
    <label><input type="checkbox" id="pgShowReal"> Overlay real samples</label>
  </div>
  <canvas id="pgCv" height="430" aria-label="Particle endpoint scatter: samples generated with the chosen solver and step count"></canvas>
  <div class="fms-stats">
    <span><span class="k">NFE</span> <b id="pgNfe"></b></span>
    <span><span class="k">On-peak fraction (within 2σ)</span> <b id="pgHqf"></b></span>
    <span><span class="k">Modes covered</span> <b id="pgModes"></b>/8</span>
  </div>
  <figcaption><b>Figure 4 · The sampling playground (interactive: change solver, drag step count, overlay real samples).</b> Scatter of the integration endpoints of 1500 particles on the exact flow field; the red × marks are the eight true mode centers. The "on-peak fraction" is the experimental metric high_quality_frac (the fraction landing within the 2σ radius of any mode) — note that the theoretical ceiling for a perfect reproduction is about 0.865, not 1 (a 2D Gaussian itself puts only 86.5% of its mass within 2σ). The Dopri5 setting uses a fixed rtol=10⁻³.</figcaption>
</figure>

<div class="ln-eyebrow">Experiments · trained model</div>

## 8 · Experiments: comparison on a genuinely trained model

Toy fields are easy to understand, but a conclusion only counts if it holds on a **genuinely trained model**. The setup: train one flow matching MLP on each of two 2D datasets, 8-Gaussians and Two-Moons (15000 steps, EMA, fixed seed); then freeze it and let five solvers sample from the **same batch** of 2000 evaluation noise vectors (a paired comparison), with NFE on the horizontal axis. Full code and data are in `fm_solvers.py` / `results.json`.

<div class="ln-chips">
  <span class="ln-chip">Two-Moons NFE to floor · Euler <b>40</b></span>
  <span class="ln-chip">Midpoint/Heun <b>16</b></span>
  <span class="ln-chip good">RK4 <b>12</b></span>
  <span class="ln-chip">8-Gaussians sampling floor W2 <b>0.165</b></span>
  <span class="ln-chip bad">At NFE=4, Euler <b>0.194</b> &lt; Heun <b>0.280</b></span>
</div>

### 8.1 · Distribution quality W2: the "floor" arrives quickly

W2 (the 2-Wasserstein distance) measures how far the generated distribution is from the real data. In Figure 5 it drops fast and then **saturates** — the remaining gap comes from model error and the sampling noise of 2000 points, and no amount of extra NFE will buy it down. The solvers differ only in *how quickly* they reach the floor.

<figure class="fms-panel">
  <div class="fms-legend" id="w2Legend"></div>
  <canvas id="w2Cv" height="620" aria-label="Line chart of W2 vs NFE on two datasets"></canvas>
  <figcaption><b>Figure 5 · W2 vs NFE on two datasets (real experimental data; interactive: hover for readouts).</b> The dashed line is the sample floor (the W2 between two independent sets of real samples, i.e. the sampling-noise floor). The 8-Gaussians floor (0.165) sits above the model's converged value, so W2 on that side saturates at NFE≈6–8 and cannot separate the solvers; the Two-Moons side shows the ordering clearly: reaching the floor takes Euler NFE≈40, Midpoint/Heun≈16, RK4≈12.</figcaption>
</figure>

### 8.2 · Pure numerical accuracy, endpoint_err: the textbook slopes appear

Take RK4 with 500 steps (NFE=2000) as the "true answer", and measure how far each solver's endpoint deviates. This metric strips out model error, leaving only discretization error — and so the textbook slopes of Section 5 reappear unchanged on a **trained network** (Figure 6).

<figure class="fms-panel">
  <div class="fms-legend" id="epLegend"></div>
  <canvas id="epCv" height="620" aria-label="Log-log line chart of endpoint error vs NFE on two datasets"></canvas>
  <figcaption><b>Figure 6 · Endpoint error vs NFE (log-log, real experimental data; interactive: hover for readouts).</b> The dashed gray lines are slope −1/−2/−4 references: Euler hugs −1, Midpoint/Heun hug −2, RK4 hugs −4 early on, and the higher-order methods eventually hit a floor of ~10⁻³ — that being the accuracy limit of the RK4-500 reference solution itself. <span class="fms-t-dopri5">★ Dopri5</span> needs no step-count tuning and naturally lands on the Pareto frontier.</figcaption>
</figure>

### 8.3 · Five conclusions worth remembering

<div class="fms-takeaways">
  <div class="fms-tk"><div class="n">1</div><p class="hd">A 3× spread in the minimum NFE to reach the floor</p>
    <p>Reaching ≤1.05× the floor on Two-Moons: Euler 40 · Midpoint 16 · Heun 16 · RK4 12. Higher-order methods buy the same distribution quality with about 1/3 of the compute.</p></div>
  <div class="fms-tk"><div class="n">2</div><p class="hd">At very low NFE, higher order does not always win</p>
    <p>W2 on 8-Gaussians at NFE=4: Euler 0.194 &lt; Heun 0.280. With too few steps, the "expensive per step" cost of a higher-order method outweighs its "accurate per step" benefit — try it yourself in Figure 4.</p></div>
  <div class="fms-tk"><div class="n">3</div><p class="hd">W2 has a floor; endpoint error never lies</p>
    <p>Distribution metrics saturate against sampling noise (the 8G floor is 0.165), and past saturation every solver looks equally good. To compare numerical accuracy you need the paired endpoint error.</p></div>
  <div class="fms-tk"><div class="n">4</div><p class="hd">Dopri5 = the tuning-free frontier</p>
    <p>Give it an rtol and it paces itself: a loose tolerance already yields decent samples at NFE≈37, a tight one converges smoothly toward the reference solution, and at no point do you sweep a grid of step counts by hand.</p></div>
  <div class="fms-tk"><div class="n">5</div><p class="hd">If the field is broken, nothing can save you</p>
    <p>The undertrained ablation (1000 training steps): the W2 floor jumps from 0.05 to 0.33+, and extra NFE does nothing; on Two-Moons Euler overtakes RK4 — a higher-order method merely integrates a <b>wrong field</b> more precisely.</p></div>
</div>

<div class="ln-eyebrow">Wrap-up · decision table</div>

## 9 · Practical guide

The whole note compressed into a single decision table:

| Solver | Order | NFE/step | When to use it |
|---|---|---|---|
| <span class="fms-chip fms-c-euler"></span>Euler | 1 | 1 | The fallback when the budget is tiny (NFE≤6) or the field is coarse (undertrained/distilled models) |
| <span class="fms-chip fms-c-midpoint"></span>Midpoint | 2 | 2 | The low-NFE sweet spot (8–16); the most reliable option at low budget in these experiments |
| <span class="fms-chip fms-c-heun"></span>Heun | 2 | 2 | Also 2nd order, with a slightly different error constant; popular in the diffusion community (the default second-order method of EDM<sup>[[9]](#refs)</sup>) |
| <span class="fms-chip fms-c-rk4"></span>RK4 | 4 | 4 | Medium-to-high budget (NFE≥12) when you want a high-fidelity endpoint; fastest to reference accuracy |
| <span class="fms-chip fms-c-dopri5"></span>Dopri5 | 5(4) | adaptive | When you do not want to tune step counts: set rtol and it lands on the Pareto frontier automatically |

One step further takes us outside the scope of this note: in real image/video generation, the low-NFE regime also hosts a family of **specialized solvers that exploit the structure of the diffusion ODE** — DDIM<sup>[[8]](#refs)</sup>, DPM-Solver++<sup>[[10]](#refs)</sup>, UniPC<sup>[[11]](#refs)</sup>, plus the "Euler + shifted schedule" actually used by flow matching models such as SD3/Flux<sup>[[12]](#refs)</sup>. For how they relate to general-purpose solvers, see the last item in the Q&A.

*Cross-reference to the code: the solver implementations live in `fm_solvers.py` as `solve_euler / solve_midpoint / solve_heun / solve_rk4 / solve_dopri5`; convergence order and NFE bookkeeping are guarded by `test_solvers.py`; the data on this page comes from `summary.csv`.*

<div class="ln-eyebrow">Appendix · Q&A</div>

## 10 · Q&A

*Real questions that came up while learning, added to over time. Click each one to see the answer.*

???+ question "Is the y-axis in the figures 'accuracy'? Is it the same thing as generation quality?"

    The y-axis of Figure 2 (Section 5) and Figure 6 (Section 8.2) is the **endpoint error**: the distance from the solver's computed endpoint to the true solution (analytic in Figure 2, the RK4-500 reference solution in Figure 6), on a log scale. It measures **pure numerical integration accuracy** — the lower it is the more accurate the integration, and the log-log slope is the order of convergence.

    But it is **not** generation quality. For quality, look at W2 in Figure 5 (Section 8.1), which compares the generated distribution against the real data. The division of labor between them is exactly conclusion 3: W2 "saturates" against sampling noise, and past saturation every solver looks equally good; endpoint error strips out model error and sampling noise, so it can always tell them apart. In one line: *W2 answers "do the samples look right", endpoint error answers "is the integration accurate".*

??? question "What exactly are rtol and atol? Why is it called a 'relative' tolerance?"

    A tolerance is how large an error you allow the solver **to make in each step**. There are two ways to state one. **atol (absolute)** — "the error must not exceed 0.0001, no matter how large the value is" — a ruler with a fixed scale. **rtol (relative)** — "the error must not exceed one thousandth of the value itself" — accuracy demanded in proportion.

    Why "relative"? Being off by 1 meter when measuring Mount Everest is irrelevant (0.01% of the total), while being off by 1 meter when measuring someone's height is absurd — **whether a given error is serious depends on how big the thing being measured is**. But a purely relative criterion goes out of control as the value approaches 0 ("one thousandth of 0 is still 0", forcing the step size toward zero), so atol provides a backstop. The formula in practice:

    $$
    \text{tolerance} = \mathrm{atol} + \mathrm{rtol}\cdot|x|
    $$

    When the value is large the rtol term dominates (proportional); as the value approaches 0 the atol term takes over (an absolute floor). These experiments use atol = rtol/100.

??? question "Explain RK4 in plain language?"

    Think of one integration step as **driving a stretch of winding road with your eyes closed**, where opening your eyes to look at the road once = one network forward pass (1 NFE).

    <span class="fms-t-euler">Euler</span>: glance at the direction before setting off, then drive the whole stretch blind — the moment the road bends, you shoot off it (the "diffuse ring" at low step counts in Figure 4 comes from exactly this).

    <span class="fms-t-rk4">RK4</span> cleverly looks four times: one glance at the start (k₁) → drive along k₁ to the **halfway point** and look again (k₂) → go back to the start, drive along k₂ to the halfway point and look once more (k₃) → drive along k₃ to the **endpoint** for one final look (k₄). Then take the weighted average (k₁+2k₂+2k₃+k₄)/6 — the two halfway glances get the largest weight, because the direction at the midpoint is the most representative of the whole stretch — and actually take the step along that "average direction".

    In essence: four samples reconstruct the **average velocity** over the whole stretch (mathematically, Simpson's rule within the step), so it can hug the curve too. The cost is 4 NFE per step; the payoff is that halving the step size drops the error 16-fold — it pays for itself quickly. Switch to RK4 in Figure 1 and drag dt up to see the four glances for yourself.

??? question "Explain Dopri5 in plain language?"

    RK4 is a good driver, but **you have to fix the step size in advance**. Dopri5 = a good driver plus adaptive cruise control, with four tricks:

    **① Seven glances per step, assembling two versions of the answer** — a fine one (5th order) and a coarse one (4th order).<br>
    **② The difference between the two versions ≈ how wrong this step is** — self-checking without knowing the true answer, and free.<br>
    **③ Adjust the speed accordingly**: the two versions nearly agree → the road is straight → step on the gas for the next step (dt up to ×5); they differ too much → **scrap this step and redo it**, tightening the stride (down to ×0.2). Like a navigation speed limit: 120 on the highway, automatically 30 entering a village — the "loose early, tight late plus red-outlined retries" of Figure 3 is exactly this at work.<br>
    **④ Tail-to-head splicing (FSAL)**: the last glance of this step happens to land on the starting point of the next, so it is reused directly, and seven glances cost the price of six.

    All you have to say is "I want accuracy of one part in a thousand" (rtol), and where to speed up and where to slow down is entirely its decision — that is what "tuning-free" means.

??? question "When actually using a diffusion / flow-matching model, is Dopri5 the first choice?"

    **No.** Dopri5 is the "safe default when you do not want to tune anything" (it is the default in SciPy's `RK45`, MATLAB's `ode45`, and torchdiffeq), but the low-NFE regime of production-grade generative models is occupied by specialized methods:

    | Scenario | Mainstream choice |
    |---|---|
    | Stable Diffusion 3 / Flux (flow matching) | Euler + shifted schedule, NFE 20–30<sup>[[12]](#refs)</sup> |
    | Low-NFE sampling for diffusion models | DPM-Solver++ / UniPC (NFE 10–20)<sup>[[10]](#refs)</sup><sup>[[11]](#refs)</sup> |
    | EDM / Karras family | Heun + a carefully designed σ grid<sup>[[9]](#refs)</sup> |
    | The origin of deterministic fast sampling | DDIM<sup>[[8]](#refs)</sup> |
    | Distilled models (LCM / Turbo / Lightning) | Euler, 1–4 steps |
    | Exact likelihood / inversion / reference solutions | **Dopri5 ✓**<sup>[[6]](#refs)</sup> |

    Four reasons production does not use it: ① at low NFE it cannot beat the specialized methods — in the data on this page even Dopri5's loosest tolerance starts at NFE≈37, while Midpoint reaches the floor at NFE=16, and rejected steps burn money for nothing on a large model. ② The DPM-Solver family exploits the **semi-linear structure** of the diffusion ODE (integrating the linear part analytically), whereas a general-purpose RK method treats the network as a black box and throws that prior away. ③ Adaptive step sizes mean the NFE per image is unpredictable, while a fixed step count gives constant latency and deploys cleanly. ④ The schedule (σ/timestep schedule) often matters more than the solver's order, and fixed-step methods let you tune it directly.

    **Where Dopri5 really shines**: an unfamiliar field, no budget for tuning, a need for high accuracy (computing likelihoods, doing inversion, generating reference solutions), an NFE budget ≥30, and no concern about variability. In one line: *the tuning-free Pareto frontier, not the overall optimum — for budget-constrained production sampling, the frontier belongs to the specialized low-NFE methods.*

## References { #refs }

1. R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud. *Neural Ordinary Differential Equations.* NeurIPS 2018.
2. Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. (probability flow ODE)
3. J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
4. Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. *Flow Matching for Generative Modeling.* ICLR 2023.
5. X. Liu, C. Gong, Q. Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023.
6. J. R. Dormand, P. J. Prince. *A family of embedded Runge–Kutta formulae.* Journal of Computational and Applied Mathematics, 6(1):19–26, 1980.
7. E. Hairer, S. P. Nørsett, G. Wanner. *Solving Ordinary Differential Equations I: Nonstiff Problems.* Springer, 2nd ed., 1993. (the standard textbook on step-size control and stiff problems)
8. J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021.
9. T. Karras, M. Aittala, T. Aila, S. Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022. (EDM, the Heun sampler and the σ grid)
10. C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, J. Zhu. *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.* NeurIPS 2022; and *DPM-Solver++*, arXiv:2211.01095.
11. W. Zhao, L. Bai, Y. Rao, J. Zhou, J. Lu. *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models.* NeurIPS 2023.
12. P. Esser, S. Kulal, A. Blattmann, et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML 2024. (Stable Diffusion 3)

*The experimental part of this note (the data in Figures 5–6, the training configuration, the metric definitions) is generated entirely by the companion code `fm_solvers.py / plots.py / test_solvers.py` and reproduces with a single command; every interactive figure on this page is real numerical integration inside the browser.*
