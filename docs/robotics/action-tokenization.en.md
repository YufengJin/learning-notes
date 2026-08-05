# Robot Action Tokenization

<div class="ln-byline">2026-08-06 · about 25 min read · Yufeng Jin</div>

This note takes the chain "continuous action → discrete token → continuous action" and unpacks the background knowledge each link requires. The main thread is the **π₀-FAST style discrete autoregressive route**<sup>[[1]](#refs)</sup>, held up against FSQ, Binning, and the continuous alternative, flow matching<sup>[[2]](#refs)</sup>.

Three labels appear throughout:

- <span class="ln-tag ln-direct">DIRECT</span> — pure mathematical transform (deterministic, no training needed)
- <span class="ln-tag ln-learned">LEARNED</span> — fitted from data (statistical fit or neural network)
- <span class="ln-tag ln-lossy">LOSSY</span> — information is lost

<div class="ln-howto"><b>How to read this</b>: for the panorama, read Section 0 and Section 7.
If you only care about one component (DCT / quantization / BPE / VQ-VAE / FSQ / binning), jump
straight to that section — each is self-contained. Sections 11-14 take the system view of fitting
the components into a model; Sections 15-16 are the cross-comparison and the full π₀-FAST picture.
A glossary sits at the end.</div>

---

<div class="ln-eyebrow">Big picture · problem landscape</div>

## 0. The big picture: what problem is actually being solved

What a robot policy has to do: **given an observation $o$ (images + proprioceptive state), output a stretch of future actions $a_{1:T}$**. Actions are **continuous** real-valued vectors (joint angles, end-effector pose, gripper opening).

There are two completely different routes to modelling $p(a_{1:T}\mid o)$:

| Route | Action representation | Representative | Analogy |
|---|---|---|---|
| **Continuous generation** | emit real-valued vectors directly | π₀ (flow matching / diffusion) | like an image diffusion model |
| **Discrete autoregression** | turn actions into tokens, predict them one by one like an LLM | π₀-FAST | like GPT writing a sentence |

This note focuses on the **second route**. The core magic: "translate" continuous actions into a string of integer tokens, and robot control **becomes a language modelling problem** — the whole LLM machinery (Transformer, cross-entropy, sampling) can be reused as is.

!!! note "The one-sentence intuition"
    Tokenization = issuing continuous actions a "dictionary" that translates smooth curves into a finite set of "words". All the difficulty lies in: how the dictionary is built (learned or direct?), how much information the translation loses (where is the lossy step?), and how faithfully it translates back (invertibility).

**Concept dependency graph:**

```text
robot action (sequence of continuous vectors)
   │  why not regress it directly? → the mode averaging problem (§2.1)
   ▼
motivation for discretization ─┐
   │                           │
   ├─ FAST route:              ├─ signal processing: DCT (§4) compacts energy
   │   DCT→quantize→BPE        ├─ quantization/rounding (§5) the one lossy step
   │                           └─ BPE (§6) lossless compression vocabulary
   │
   ├─ FSQ route: neural autoencoder + VQ/FSQ codebook (§8, §9)
   │
   └─ Binning route: uniform bucketing (§10, the most naive baseline)
                ▼
        all of them feed an autoregressive Transformer (§11)
        trained with cross-entropy → naturally multimodal (§12)
        sampling + temperature (§12.1) pulls out different solutions
        attention masking (§13) separates prompt from action
```

---

<div class="ln-eyebrow">Prerequisites</div>

## Prerequisites: the core concept triples

A few concepts recur throughout the note; here they are as **English term = Chinese term = minimal definition**. The Chinese names are kept as a bridge back to the Chinese literature, and the wording in the body matches these entries.

- **tokenization = 词元化** = mapping text or a continuous signal onto a sequence of integer IDs drawn from a finite vocabulary, so a Transformer can treat them as categories<sup>[[3]](#refs)</sup>.
- **mode averaging = 模式平均** = the phenomenon where L2/MSE regression averages several mutually exclusive correct solutions into one wrong solution; it is the central motivation for this entire discretization route (§2.1).
- **DCT (discrete cosine transform) = 离散余弦变换** = an orthogonal transform that invertibly decomposes a discrete signal into the weights (coefficients) of a set of cosine bases at different frequencies; for a smooth signal the energy concentrates in the low-frequency coefficients<sup>[[4]](#refs)</sup>.
- **quantization = 量化** = the operation of mapping continuous values onto a finite set of grid points (e.g. multiply by a scale, then round)<sup>[[5]](#refs)</sup>; it is the only lossy step in the "continuous → discrete" chain (§5).
- **BPE (byte pair encoding) = 字节对编码** = a lossless compression algorithm that repeatedly merges the most frequent adjacent pair in a sequence to shorten it; the merge rules come from data statistics and involve no neural network<sup>[[3]](#refs)</sup>.
- **prefix-LM = 前缀语言模型** = a mixed-mask Transformer in which the prefix (prompt/condition) uses bidirectional attention while the suffix (generation target) uses causal attention<sup>[[6]](#refs)</sup>.

---

<div class="ln-eyebrow">Basics 01 · what an action is</div>

## 1. What a robot action is (the object being tokenized)

One action sample is a two-dimensional matrix:

```text
actions.shape == (T, D)
            ┌─ T = action_horizon: how many future steps one prediction covers (e.g. 50 steps ≈ 1 second @ 50Hz)
            └─ D = action_dim: degrees of freedom per step (e.g. 7 = 6 joints + 1 gripper, or 14 = dual arm)
```

Two key properties:

- **① Smooth in time**: consecutive actions differ little (robots do not teleport) → this is the precondition that lets DCT compress.
- **② Different units per dimension**: joint angles may span ±3 rad, the gripper 0~1 → they must first be **normalized** onto a common range.

!!! tip "Why a 'stretch' rather than a single step"
    Predicting a whole **action chunk** (instead of a single step) reduces compounding error and makes motion more coherent; it also hands DCT a "time axis" on which to do a frequency-domain transform. A single-step action has no frequency domain to speak of.

### Normalization: the first preprocessing step <span class="ln-tag ln-learned">LEARNED</span> (statistical quantiles, approximately lossless)

Scale every action dimension to roughly $[-1,1]$. The usual choice is **quantile normalization** rather than plain min-max: take the 1% and 99% quantiles as the bounds, which is more robust to outliers.

```python
# Idea: use quantiles computed on the training set to squeeze actions into [-1,1]
a_norm = 2 * (a - q01) / (q99 - q01) - 1   # anything outside gets clipped
```

!!! note "Why this counts as LEARNED even though it is not a neural network"
    "Learned" here = **parameters fitted from data statistics** (the quantiles), not weights trained by gradient descent. This distinction runs through the whole note: FAST's BPE vocabulary and the normalization quantiles are both this kind of "statistical fit", **not a neural network**.

---

<div class="ln-eyebrow">Basics 02 · why discretize</div>

## 2. Why discretize continuous actions at all

The most natural idea is to have the network output real-valued actions directly, supervised by L2/MSE regression. So why go to the trouble of discretizing into tokens? Three reasons:

1. **Reuse the entire LLM infrastructure**: once actions are tokens, you can plug straight into a pretrained VLM's (e.g. PaliGemma) Transformer, vocabulary and cross-entropy, and skip designing a new output head.
2. **Multimodality comes for free** (detailed in §12): a discrete distribution can assign high probability to "go left" and "go right" simultaneously; regression cannot.
3. **Compression + efficient autoregression**: FAST uses DCT+BPE to squeeze a stretch of actions into a very short token sequence, making autoregressive inference faster.

### 2.1 The core motivation: the mode averaging disaster

This is **the single most important intuition for understanding the whole discretization route**. Picture an obstacle in front of the robot with two equally correct ways around it: **go around the left** or **go around the right**. Both appear in the training data.

!!! danger "The fatal flaw of L2 regression"
    MSE regression is essentially fitting a **unimodal Gaussian**, and its optimum is the **average** of all correct answers. The average of "left" and "right" = **driving straight into the obstacle**. That is mode averaging.

<div class="ln-demo">
<div class="ln-demo-title">Figure 1 · mode averaging vs a multimodal distribution (interactive demo)</div>
<div class="ln-demo-hint">The data contains two correct actions (blue cluster = go left, green cluster = go right). Drag the slider to change the distance between the clusters and watch where the L2 regression's "optimal prediction" (red line) lands. Takeaway: the red line always falls in the no-man's-land between the clusters — regression averages two correct solutions into one wrong one, which is mode averaging; tick "show discrete distribution" to see a multimodal distribution keep both solutions alive.</div>
<canvas id="cvMode" width="900" height="280"></canvas>
<div class="ln-controls">
<label>Gap between the two modes <input type="range" id="modeGap" min="0" max="100" value="70"></label>
<label><input type="checkbox" id="showCat" checked> Show the discrete distribution (multimodal)</label>
</div>
<div class="ln-readout" id="modeOut"></div>
</div>

The wider the gap, the deeper the L2 "optimal prediction" (the mean of the two clusters) sinks into **no-man's-land** — an action nobody ever demonstrated, one that may drive straight into a wall. A discrete distribution, by contrast, can light up **both** the left and the right bin with high probability, preserving both modes perfectly. That is why continuous actions get sliced into discrete tokens and trained with classification cross-entropy.

---

<div class="ln-eyebrow">Basics 03 · tokenization</div>

## 3. The tokenization concept (borrowed from NLP)

In NLP, tokenization = cutting a continuous stream of characters into integer IDs from a finite vocabulary:

```text
"the robot is cute"  →  ["the","robot","is","cute"]  →  [8123, 442, 19, 5601]
       text                    subwords                    integer token ids
```

These integers then index an **embedding table** and become vectors fed into the Transformer. The key points:

- The **vocabulary** is finite (PaliGemma, for instance, has 257,152 tokens<sup>[[6]](#refs)</sup>).
- Each token id corresponds to one row of embedding vector in the vocabulary.
- The model's output is a **probability distribution over the vocabulary** (softmax over 257,152 classes).

!!! note "The core trick for carrying this over to robots"
    The last **128** ids of the PaliGemma vocabulary are almost never used in practice. You can **commandeer those 128 slots** as action tokens — so action tokens and text tokens **share one embedding table and one softmax output head**. Robot actions really are stuffed into the language model as "words of a new language".

---

<div class="ln-eyebrow">Component 01 · DCT</div>

## 4. DCT, the discrete cosine transform <span class="ln-tag ln-direct">DIRECT</span>

This is FAST's first step, and the piece most people are least familiar with.

### Intuition: decompose a curve into a sum of cosine waves at different frequencies

Any discrete signal (say the 50 samples of one joint angle over time) can be represented **exactly** as a weighted sum of cosine waves at various frequencies. DCT is precisely the computation of "how much weight each frequency carries"<sup>[[4]](#refs)</sup>.

!!! tip "Analogy: a mixing desk"
    Picture an audio equalizer with one fader each for low (bass), mid and high frequencies. DCT decomposes the signal onto those "frequency faders". Robot actions are **smooth** → almost all low frequency → only the leftmost few faders carry any value, and a wide stretch on the right sits at ≈0. That is "energy concentrated in the low frequencies".

### The formula (DCT-II; knowing what it looks like is enough)

Turning a length-$N$ signal $x_0,\dots,x_{N-1}$ into coefficients $X_0,\dots,X_{N-1}$:

$$
X_k = \sum_{n=0}^{N-1} x_n \cos\!\left[\frac{\pi}{N}\left(n+\tfrac12\right)k\right]
$$

$k=0$ is the DC component (the overall mean); larger $k$ means higher frequency. The inverse transform IDCT adds the coefficients back up using the same cosine bases and is **fully invertible** (up to floating-point error). The DCT is taken along the **time axis T**, independently for each action dimension D.

<div class="ln-demo">
<div class="ln-demo-title">Figure 2 · DCT energy compaction (interactive demo)</div>
<div class="ln-demo-hint">The top plot is a stretch of "action signal" (with adjustable smoothness); the bottom plot shows its DCT coefficients, low frequencies at the left of the horizontal axis and high frequencies at the right. Watch how a smooth signal crams its energy into the leftmost few low-frequency coefficients. Takeaway: the smoother the signal, the more tightly the non-zero coefficients cluster at the low-frequency end — this is energy compaction, and it is what makes the later rounding and BPE compression so efficient.</div>
<canvas id="cvDct" width="900" height="360"></canvas>
<div class="ln-controls">
<label>Signal smoothness <input type="range" id="dctSmooth" min="1" max="20" value="3"></label>
<button id="dctReroll">🎲 New signal</button>
</div>
<div class="ln-readout" id="dctOut"></div>
</div>

!!! note "Why DCT is discretization's 'perfect assist'"
    DCT itself **does not discretize and loses no information** (it is pure invertible mathematics). What it does is **redistribute** the signal's energy: a smooth action → most coefficients ≈0. At the next step, rounding crushes those small coefficients close to 0 into **exactly** 0, producing a mass of repeated 0s → superb raw material for BPE. **DCT does not compress, but it makes the compression that follows efficient.**

---

<div class="ln-eyebrow">Component 02 · quantization</div>

## 5. Quantization / rounding <span class="ln-tag ln-direct">DIRECT</span> <span class="ln-tag ln-lossy">LOSSY</span>

This is **the only step in the whole FAST chain that genuinely loses information**, and the place where "continuous → discrete" actually happens.

A continuous value can take infinitely many values; a token vocabulary can only express finitely many. Quantization = **mapping a continuous value onto the nearest "grid point"**<sup>[[5]](#refs)</sup>. In its simplest form: multiply by a scaling factor, then round.

```python
q = round(x * scale)        # continuous x → integer q   (encode, lossy)
x_hat = q / scale           # integer q → approximate continuous x̂ (decode, the original value is gone)
```

The error $|x-\hat x|$ is at most about $\frac{1}{2\,\text{scale}}$. A larger scale → a denser grid → more precision, but a wider integer range and more tokens. This is the **precision vs compression ratio** trade-off.

<div class="ln-demo">
<div class="ln-demo-title">Figure 3 · The precision / compression trade-off in quantization (interactive demo)</div>
<div class="ln-demo-hint">The blue line is the original continuous signal; the red staircase is the signal reconstructed after quantization. Drag "quantization levels" and see: fewer levels are cheaper (fewer tokens) but more distorted. Takeaway: the number of quantization levels is exactly the precision vs compression knob — fewer levels means a coarser staircase and larger reconstruction error; this step is also the only lossy link in the entire chain.</div>
<canvas id="cvQuant" width="900" height="300"></canvas>
<div class="ln-controls">
<label>Quantization levels (bins) <input type="range" id="qBins" min="2" max="64" value="8"> <span class="ln-val" id="qBinsV">8</span></label>
</div>
<div class="ln-readout" id="quantOut"></div>
</div>

!!! warning "The subtlety of FAST"
    FAST does not quantize the raw action; it quantizes the **DCT coefficients**<sup>[[1]](#refs)</sup>. Because the high-frequency coefficients are ≈0, rounding turns them into a broad field of 0s — affordable to lose (high frequencies contribute almost nothing to a smooth action) while manufacturing a compressible repeated pattern. "Rounding in the frequency domain" is far smarter than "rounding in the time domain".

---

<div class="ln-eyebrow">Component 03 · BPE</div>

## 6. BPE, byte pair encoding <span class="ln-tag ln-learned">LEARNED</span> (statistical fit, not an NN, lossless)

Rounding yields a long string of integers (with plenty of repetition, especially 0s). BPE is a **lossless compression** algorithm<sup>[[3]](#refs)</sup> that merges "frequently repeated patterns" into a single new token, shortening the sequence.

Repeat the following: "find the **most frequent adjacent pair** in the current sequence and merge it into a new symbol", until the target vocabulary size is reached.

```text
initial: a a b a a b a a b       (the pair a a occurs often)
merge (a,a)→Z:  Z b Z b Z b      (now the pair Z b occurs often)
merge (Z,b)→Y:  Y Y Y             ← 9 symbols squeezed into 3, fully recoverable
```

<div class="ln-demo">
<div class="ln-demo-title">Figure 4 · BPE merge compression (interactive demo)</div>
<div class="ln-demo-hint">Below is a stretch of quantized integers (with lots of repetition). Click "merge once" to run one BPE step and watch the sequence shrink and the dictionary grow. Takeaway: each merge shortens the sequence and adds one merge rule to the dictionary; the whole process is invertible and the compression entirely lossless — the loss happened at the previous step, the rounding.</div>
<canvas id="cvBpe" width="900" height="170"></canvas>
<div class="ln-controls">
<button id="bpeStep">▶ Merge the most frequent pair</button>
<button id="bpeReset">↺ Reset</button>
</div>
<div class="ln-readout" id="bpeOut"></div>
</div>

"Which pairs to merge, and in what order" = the **BPE vocabulary / merge rules**, which are **derived from statistics over the training data** (hence learned), but are **only lookup-style statistical rules, containing no neural network or gradients whatsoever**. Given the vocabulary, both encoding and decoding are **entirely lossless**.

!!! note "Where the name FAST comes from"
    FAST = **F**requency-space **A**ction **S**equence **T**okenization<sup>[[1]](#refs)</sup>. DCT+BPE is the algorithm itself — pure signal processing plus statistical compression, no neural network.

---

<div class="ln-eyebrow">System 01 · FAST panorama</div>

## 7. Putting FAST together (the panorama)

The encoding direction (continuous → token):

```text
normalized action matrix A (T, D), roughly [-1,1]
   │
   ① DCT (along the time axis)     DIRECT lossless, energy pushed into low frequencies
   ▼
frequency coefficient matrix C (T, D), high frequencies ≈0
   │
   ② Scale + Round                 DIRECT · LOSSY ← discretization happens here
   ▼
sparse integer matrix (mostly 0)
   │
   ③ Flatten to 1D                 DIRECT
   │
   ④ BPE merges repeated patterns  LEARNED vocabulary · lossless
   ▼
final discrete token sequence (variable length, usually far shorter than T×D)
   │
   ⑤ map onto the last 128 slots of the PaliGemma vocabulary   DIRECT pure arithmetic
```

The decoding direction is simply **the exact reverse**: BPE decode → reshape → divide by the scale → IDCT → denormalize.

!!! danger "The conclusion people most often garble"
    **FAST is not a neural network!** It is "deterministic invertible compression (DCT + rounding + flatten) plus one statistically fitted BPE vocabulary". The only information loss comes from the rounding in step ②. That is exactly its selling point over VQ/FSQ — **no neural autoencoder to train**, usable out of the box<sup>[[1]](#refs)</sup>.

<div class="ln-demo">
<div class="ln-demo-title">Figure 5 · The full FAST round trip (continuous ↔ discrete, adjustable scale, interactive demo)</div>
<div class="ln-demo-hint">A smooth multi-dimensional action trajectory goes through DCT along the time axis → quantization by rounding → IDCT reconstruction. Drag the quantization scale and watch all four panels change at once — this is the most direct demonstration of "how continuous becomes discrete, where the loss comes from, and how many tokens result". Takeaway: the larger the scale, the more closely the reconstruction hugs the original trajectory, but the wider the magnitude range of the integer coefficients grows; the only loss along the whole chain comes from the rounding step.</div>
<canvas id="cvRT" width="900" height="250"></canvas>
<canvas id="cvTrade" width="900" height="200" style="margin-top:14px"></canvas>
<div class="ln-controls">
<label>Quantization scale (larger = finer) <input type="range" id="rtScale" min="1" max="64" value="10" step="0.5"> <span class="ln-val" id="rtScaleV">10</span></label>
<button id="rtReroll">🎲 New trajectory</button>
</div>
<div class="ln-readout" id="rtOut"></div>
</div>

### 7.1 A complete numerical example: every step's shape and data made visible

Many people have no concrete feel for "what shape actually goes into and comes out of the DCT, and how scale+round plays out". Below is a walk-through with **genuinely computed numbers** (for clarity we take `D=1`, a single dimension, and `T=8` time steps; with more dimensions each column independently repeats the same process).

**Input: the normalized action matrix A**, of shape `(T, D) = (8, 1)`, values already normalized to $[-1,1]$:

```text
A = [ 1.000, 0.816, 0.494, 0.111, -0.258, -0.555, -0.755, -0.853 ]   # shape (8,1)
     t=0    t=1    t=2    t=3    t=4     t=5     t=6     t=7         ← time axis
This is a smoothly descending curve (what a robot action typically looks like).
```

**① DCT — transform along the time axis; input and output shapes are identical** (`(T,)→(T,)`, and for `(T,D)` it is one transform per column):

```text
C = DCT(A) = [ 0.00, 1.889, 0.159, -0.00, -0.00, 0.00, 0.00, -0.00 ]   # still (8,1)
              k=0   k=1    k=2    k=3    k=4   k=5   k=6   k=7         ← frequency axis
              DC    low ────────────→                     high
Nearly all the energy sits at k=1 (=1.889)! The high-frequency coefficients from k≥3 are already ≈0.
```

!!! tip "The key realization"
    DCT **does not change the number of elements**: 8 numbers in, 8 numbers out. All it does is **change coordinates** — from "time" to "frequency". Not a scrap of information is lost (it is invertible), but it is **redistributed** onto a handful of low-frequency coefficients.

**② Scale + Round — discretization happens here** (with `scale = 8`):

```text
C × scale = [ 0.00, 15.115, 1.273, -0.00, -0.00, 0.00, 0.00, -0.00 ]
round(·)  = [   0,    15,     1,     0,     0,    0,    0,    0    ]   = Cq (integers!)
                                                                       only 2 of the 8 are non-zero
```

This step: ① turns continuous decimals into **integers** (the real "discretization"); ② crushes those ≈0 high-frequency coefficients into **exactly** 0. The loss originates here (the rounding of `15.115→15` and `1.273→1`).

**③ Flatten** flattens `(T,D)` into the 1D integer sequence `[0, 15, 1, 0, 0, 0, 0, 0]` (for `D>1` it is straightened out along `T×D`).

**④ BPE** merges frequently repeated patterns (such as "5 consecutive 0s") into single tokens, shortening the sequence, and finally **⑤** maps them onto the ids of the top 128 slots of the vocabulary.

**Decoding: retrace the path and see how small the loss is**

```text
token → BPE decode → [0,15,1,0,0,0,0,0] → reshape(8,1) → ÷scale → IDCT → denormalize
reconstructed A_rec = [0.977, 0.803, 0.497, 0.125, -0.241, -0.545, -0.756, -0.862]
original A          = [1.000, 0.816, 0.494, 0.111, -0.258, -0.555, -0.755, -0.853]
reconstruction MSE ≈ 1.7e-4   ← 8 values recovered almost perfectly from just 2 non-zero coefficients!
```

!!! note "This example says it all"
    8 continuous values → DCT → only **2** effective (non-zero) integer coefficients → a reconstruction error of just 1.7e-4. **That is the entire secret of FAST's efficient compression**: after the DCT a smooth action has extremely concentrated energy, rounding manufactures a mass of 0s, and BPE then compresses the runs of 0s away. For a deeper dive into the mathematics of DCT energy compaction, see [Fourier and DCT · energy compaction](../math/fourier-dct.md#7-dct).

---

<div class="ln-eyebrow">Component 04 · VQ-VAE</div>

## 8. VQ-VAE and codebooks (the prerequisite for understanding FSQ) <span class="ln-tag ln-learned">LEARNED</span> (a genuine neural network)

The FSQ route is **a genuine neural network approach**. To understand FSQ, first understand the "codebook" idea of its predecessor, VQ-VAE.

The idea of VQ-VAE (Vector Quantized VAE)<sup>[[7]](#refs)</sup>: maintain a **learnable "codebook"** — say 256 vectors, each called a codeword, numbered 0~255.

```text
encoder (NN) → continuous vector z
         ↓ find the nearest codeword in the codebook
index of the nearest codeword = token (e.g. #42)      ← discretization
         ↓
decoder (NN) reconstructs the action from the codeword vector of #42
```

The encoder, the decoder and **the 256 vectors of the codebook themselves** are all obtained by **gradient training** (with a reconstruction MSE loss). This is a genuine "end-to-end learned discrete representation".

!!! warning "The pain points of VQ-VAE"
    ① **Codebook collapse**: during training many codewords are never used, wasting capacity and making training hard.

    ② Taking the nearest neighbour is not differentiable → a **straight-through estimator** trick is needed to backpropagate at all.

    These pain points are exactly what **FSQ** sets out to solve.

### The straight-through estimator

"Nearest neighbour / rounding" is a staircase function whose derivative is 0 everywhere, so gradients never make it back to the encoder. The trick: **use the discrete value in the forward pass, and pretend it is the identity function in the backward pass** (let the gradient pass straight through)<sup>[[8]](#refs)</sup>.

```python
z_q = z + stop_gradient(quantize(z) - z)
#   forward:  z_q == quantize(z) (discrete)
#   backward: d z_q / d z == 1   (the gradient flows straight through as if nothing were quantized)
```

---

<div class="ln-eyebrow">Component 05 · FSQ</div>

## 9. FSQ, finite scalar quantization <span class="ln-tag ln-learned">LEARNED</span>

FSQ (Finite Scalar Quantization)<sup>[[9]](#refs)</sup> is the elegant replacement for VQ-VAE: **drop the learnable codebook and instead round onto a fixed grid, independently in each of a handful of dimensions**. It needs no codebook learning, yet is inherently collapse-free.

The core trick:

1. The encoder projects the action into a **very low-dimensional** space (say only 3 dimensions).
2. Each dimension is squashed to $[-1,1]$ with $\tanh$, then **rounded to the few fixed grid points of that dimension** (e.g. 8 grid points for dimension 1, 6 for dimension 2, 5 for dimension 3).
3. The combination of grid points across dimensions is encoded into **a single integer** token using **mixed-radix** arithmetic ($8\times6\times5=240\approx256$ codewords).

```python
# FSQ encode (conceptual)
x = proj_down(z)                      # project down to ~3 dims        [LEARNED]
zc = tanh(x)                          # squash to [-1,1]
digits = round((zc+1)*(bases-1)/2)    # round each dim to a grid point [DIRECT, LOSSY]
token = undigitize(digits)            # multi-dim digits → one integer (mixed radix) [DIRECT]
```

!!! tip "FSQ vs VQ in one line"
    VQ: the codebook is **a pile of arbitrary vectors that must be learned**, with collapse to guard against.

    FSQ: the codebook is **a fixed regular grid** (the Cartesian product of equally spaced grid points per dimension) — nothing to learn, nothing to collapse. Only the projection encoder/decoder are neural networks.

<div class="ln-demo">
<div class="ln-demo-title">Figure 6 · FSQ's fixed 2D grid codebook (interactive demo)</div>
<div class="ln-demo-hint">Once the action has been squeezed into 2 dimensions, FSQ's "codewords" are exactly these regular grid points. Move the mouse (or drag) and watch a continuous point snap to the nearest grid codeword (= one integer token). Adjust the number of grid points per dimension to see the codebook size change. Takeaway: FSQ's codebook is the fixed regular grid itself — nothing to learn, inherently collapse-free, with codebook size = the product of the grid counts across dimensions.</div>
<canvas id="cvFsq" width="420" height="420" style="margin:auto"></canvas>
<div class="ln-controls">
<label>Grid points, dim 1 <input type="range" id="fsqB1" min="2" max="9" value="6"> <span class="ln-val" id="fsqB1V">6</span></label>
<label>Grid points, dim 2 <input type="range" id="fsqB2" min="2" max="9" value="5"> <span class="ln-val" id="fsqB2V">5</span></label>
</div>
<div class="ln-readout" id="fsqOut"></div>
</div>

---

<div class="ln-eyebrow">Component 06 · binning</div>

## 10. Binning, uniform bucketing (the most naive baseline) <span class="ln-tag ln-direct">DIRECT</span>

The RT-2<sup>[[10]](#refs)</sup> / OpenVLA<sup>[[11]](#refs)</sup> style, with **zero learning**: for every dimension and every time step **independently**, cut $[-1,1]$ into 256 uniform buckets and round to the bucket index.

```python
token = round((a+1)/2 * n_bins)     # encode
a_hat = token / n_bins * 2 - 1      # decode
```

- **Upside**: simple, needs no training at all, fully invertible (up to rounding).
- **Downside**: the token count = T×D (no compression, very long sequences); precision is capped at 256 levels and it **exploits no temporal correlation**. It is the control baseline in the papers.

Seeing all three side by side:

| | Binning | FAST | FSQ |
|---|---|---|---|
| continuous→discrete | uniform bucketing + rounding | DCT→round→BPE | Transformer→FSQ quantization |
| is it an NN | ❌ | ❌ (statistical fit) | ✅ |
| needs pretraining | ❌ | only the BPE vocabulary / quantiles | ✅ needs a checkpoint |
| token count | many (T×D) | few (compressed) | medium (fixed) |
| exploits temporal correlation | ❌ | ✅ (DCT) | ✅ (attention) |

---

<div class="ln-eyebrow">System 02 · autoregressive modelling</div>

## 11. Autoregressive modelling + cross-entropy

Whichever tokenizer is used, once the actions are tokens the **training objective is exactly GPT's**: predict the next token.

What the sequence looks like:

```text
Task: pick up the cup, State: <discretized proprioceptive state>;
Action: <a₁><a₂>...<aₙ>|
└──────── prefix (the condition, no loss) ────────┘└─ postfix (loss here) ─┘
```

Note: even the proprioceptive **state is discretized** (cut into 256 buckets) and stuffed into the prompt.

The training objective = next-token cross-entropy (negative log-likelihood):

$$
\mathcal{L} = -\frac{1}{|\text{mask}|}\sum_{t}\; \text{mask}_t \cdot \log p_\theta(a_t \mid a_{<t}, o)
$$

```python
targets = one_hot(tokens[:, 1:], vocab=257152)   # shift right by one to form the target
logp    = log_softmax(logits)
token_logp = sum(targets * logp, axis=-1)        # log probability of the correct token
loss = -sum(token_logp * loss_mask) / sum(loss_mask)  # loss only over the action segment
```

!!! note "Three points to note"
    ① **There is no MSE/regression term whatsoever** — pure cross-entropy.

    ② `loss_mask` is True only over the postfix `Action:...|` → loss is computed on action tokens only; prompt/state serve purely as conditions.

    ③ At inference, tokens are sampled autoregressively one by one, stopping at `|` or EOS, after which the tokenizer transforms them back into continuous actions.

---

<div class="ln-eyebrow">System 03 · multimodality</div>

## 12. How multimodality is obtained "for free"

Back to the mode averaging problem of §2.1. Discrete tokens + cross-entropy solve it **on three levels**:

1. **A discrete categorical is inherently multimodal**: at each token position the model emits a softmax over the whole vocabulary, and can assign high probability to "the token corresponding to left" and "the token corresponding to right" at the same time. No more forced averaging.
2. **The autoregressive factorization expresses joint multimodality**: $p(a_{1:n}\mid o)=\prod_t p(a_t\mid a_{<t},o)$. A product of categoricals can express complex multimodal joint distributions — "left" is one coherent token sequence, "right" is another.
3. **At inference, sampling pulls out the different solutions** (next section).

### 12.1 Sampling and temperature

Given a multimodal distribution, how do you extract an action? Sample each token's logits with a temperature:

$$
p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}
$$

- $\tau=0$: take the argmax → deterministic, the single most likely solution.
- $\tau>0$: sample randomly from the distribution → different RNG yields different feasible solutions; the larger $\tau$, the more diverse.

<div class="ln-demo">
<div class="ln-demo-title">Figure 7 · How temperature modulates multimodal sampling (interactive demo)</div>
<div class="ln-demo-hint">Bimodal logits at one token position (two high-probability regions, "left" and "right"). Drag the temperature and watch the softmax distribution change; click "sample 20 times" to see the distribution of tokens actually drawn. Takeaway: at small τ the distribution collapses to a single spike (only the tallest peak's solution survives), while at large τ both peaks stand a chance of being drawn — the different solutions of a multimodal distribution are precisely what random sampling extracts.</div>
<canvas id="cvTemp" width="900" height="280"></canvas>
<div class="ln-controls">
<label>temperature τ <input type="range" id="tempT" min="0" max="200" value="100"> <span class="ln-val" id="tempTV">1.00</span></label>
<button id="tempSample">🎲 Sample 20 times</button>
</div>
<div class="ln-readout" id="tempOut"></div>
</div>

!!! tip "The intuition"
    τ→0: the distribution becomes a needle (only the tallest peak is taken, the other solution is discarded). Large τ: the distribution flattens and both peaks stand a chance of being drawn → left this time, right the next. **The "solutions" of a multimodal problem are pulled out by random sampling from the multimodal distribution that was preserved.**

---

<div class="ln-eyebrow">System 04 · attention masking</div>

## 13. Attention masking: bidirectional prefix vs causal action

Within the sequence, the prompt/state (prefix) and the action (postfix) play different roles and get different attention masks (`ar_mask`):

- **Prefix = bidirectional attention (like an encoder)**: the prompt + state are known conditions and can all see each other (BERT style), letting the model fully absorb the task context.
- **Postfix = causal attention (like a decoder)**: an action token may only see what lies to its **left** (GPT style), which guarantees no peeking at the future during autoregressive generation.

This "bidirectional prefix + causal postfix" mixed mask is the **prefix-LM** structure, used by both PaliGemma<sup>[[6]](#refs)</sup> and π₀-FAST<sup>[[1]](#refs)</sup>.

---

<div class="ln-eyebrow">Contrast 01 · flow matching</div>

## 14. A contrast: the other route, flow matching (π₀)

To place the discrete route on a map, let us glance at the continuous one. π₀<sup>[[2]](#refs)</sup> **does not tokenize**; it uses **flow matching**<sup>[[12]](#refs)</sup> (of the diffusion family) to generate continuous actions directly: starting from Gaussian noise, it integrates along a learned "velocity field", gradually "flowing" the noise into an action.

| | π₀-FAST (this note's main thread) | π₀ (flow matching) |
|---|---|---|
| action representation | discrete tokens | continuous vectors |
| modelling | autoregression + cross-entropy | velocity field + integration |
| source of multimodality | multimodal categorical + sampling | stochastic integration paths from noise |
| reuses the LLM | ✅ vocabulary/Transformer reused directly | needs a dedicated flow head |
| inference | token by token (speed depends on token count) | a few integration steps |

!!! note "One problem, two answers"
    Both routes are solving the multimodality puzzle of "one observation admits several reasonable actions". FAST answers with "discretize + classify", π₀ with "continuous stochastic generation". Once you understand why the discrete route needs tokenization (mode averaging + LLM reuse), you understand the central tension of this field.

---

<div class="ln-eyebrow">Contrast 02 · reconstruction</div>

## 15. Reconstruction quality of the three tokenizers compared

"Continuous → token → continuous" is necessarily lossy (because of the quantization rounding). The three tokenizers strike different balances between **reconstruction error** and **token count**. First a measured comparison, then a chance to turn the knobs yourself.

!!! warning "A point of fact to clear up"
    In practice **usually only FAST actually encodes actions into tokens**; Binning / FSQ serve more as **inference baselines** (responsible only for decoding existing tokens back into actions). The "encode-decode round trip comparison" below is a conceptual measurement made **according to each algorithm's mathematical definition**, meant to convey the precision/compression trade-offs among the three.

<div class="ln-demo">
<div class="ln-demo-title">Figure 8 · FAST vs Binning vs FSQ reconstruction round-trip comparison (interactive demo)</div>
<div class="ln-demo-hint">The same stretch of action (with adjustable smoothness/noise), encoded and then decoded by each of the three methods. The top plot overlays the three reconstructed curves (colour-coded by method, matching the bar colours below); the bottom plot compares "reconstruction MSE" and "token count" as bars. FSQ is <b>illustrative</b> (there is no trained checkpoint, so its behaviour is approximated by "fixed token count + grid quantization"). Takeaway: the smoother the signal, the lower the reconstruction error FAST achieves with fewer tokens; Binning's error does not improve with smoothness and it uses the most tokens — whether temporal correlation is exploited is the decisive difference among the three.</div>
<canvas id="cvCmp" width="900" height="250"></canvas>
<canvas id="cvCmpBar" width="900" height="190" style="margin-top:14px"></canvas>
<div class="ln-controls">
<label>Signal smoothness <input type="range" id="cmpSmooth" min="1" max="12" value="2"> <span class="ln-val" id="cmpSmoothV">2</span></label>
<label><input type="checkbox" id="cmpNoise"> Add noise</label>
<label>FAST scale <input type="range" id="cmpScale" min="2" max="40" value="12"> <span class="ln-val" id="cmpScaleV">12</span></label>
<label>FSQ token count <input type="range" id="cmpFsqN" min="4" max="24" value="12"> <span class="ln-val" id="cmpFsqNV">12</span></label>
</div>
<div class="ln-readout" id="cmpOut"></div>
</div>

| | FAST | FSQ | Binning |
|---|---|---|---|
| source of reconstruction error | rounding of DCT coefficients | FSQ grid quantization + neural reconstruction error | independent 256-bucket rounding per scalar |
| error improves with smoothness | ✅ smoother is more accurate (DCT exploits temporal correlation) | ✅ the encoder can learn temporal structure | ❌ independent of smoothness, a fixed quantization floor |
| token count | few and **variable** (compressed, ≪ T×D) | medium and **fixed** (=num_tokens) | many and fixed (=T×D, one per scalar) |
| precision at an equal token budget | high (for smooth signals) | high (if well trained) | low (wasted on high frequencies) |
| training required | only the BPE vocabulary / quantiles (statistical fit) | ✅ needs a trained neural checkpoint | ❌ zero training |

!!! tip "How to read this table"
    The core trade-off is **"token count ↔ precision ↔ whether temporal structure is exploited"**. Binning is the most naive: one token per scalar, constant error but the most tokens, and no notion of temporal correlation. FAST uses DCT to turn temporal correlation into low-frequency sparsity, **buying higher precision with fewer tokens** (provided the action is smooth). FSQ uses a neural network to learn a compact latent space, with a fixed and controllable token count, but it has to be trained first.

---

<div class="ln-eyebrow">System 05 · full model</div>

## 16. The full π₀-FAST model: base model / inputs / outputs

Once the tokenizer has turned actions into tokens, who consumes those tokens? This section lays out the π₀-FAST model skeleton.

### 16.1 The base model: PaliGemma (a Gemma-2B decoder + SigLIP vision)

The backbone of π₀-FAST is **PaliGemma**<sup>[[6]](#refs)</sup> — a vision-language model (VLM) made of two parts:

| Component | What it is | Specification |
|---|---|---|
| vision encoder | **SigLIP So400m/14** (ViT)<sup>[[13]](#refs)</sup> | patch=14, 224×224 → 16×16 = **256 image tokens per image** |
| language model (LLM) | **Gemma-2B, decoder-only** | width=2048, depth=18, heads=8, vocab=**257,152** |

!!! note "Is it decoder-only? — yes, but with a prefix-LM mask"
    Gemma itself is a **decoder-only** language model. But π₀-FAST uses **attention masking** to turn it into a **prefix-LM**: the prefix of image + prompt + state attends **bidirectionally**, while the action postfix attends **causally**. So both statements — "decoder-only architecture" and "prefix-LM mask" — are correct. For the masking details and an interactive comparison, see [Autoregressive models · attention masking](../ml/autoregressive-models.md#2).

### 16.2 Inputs and the multimodal embedding order

The model input contains: images from several cameras, their respective masks, the proprioceptive state, and the tokenized text prompt. They are concatenated into **one token sequence**, in this order:

<div class="ln-fig">
<svg viewBox="0 0 860 230" role="img" aria-label="π0-FAST input sequence structure">
  <text x="10" y="20" fill="#6B675F" font-size="13">Input token sequence (concatenated along the sequence axis): left→right</text>
  <rect x="10" y="36" width="150" height="56" rx="8" fill="rgba(63,185,80,.13)" stroke="#3fb950"/>
  <text x="85" y="60" text-anchor="middle" fill="#3fb950" font-size="12" font-weight="700">① image tokens</text>
  <text x="85" y="78" text-anchor="middle" fill="#6B675F" font-size="10">base_0 / base_1 / wrist_0</text>
  <rect x="166" y="36" width="330" height="56" rx="8" fill="rgba(63,185,80,.13)" stroke="#3fb950"/>
  <text x="331" y="58" text-anchor="middle" fill="#3fb950" font-size="12" font-weight="700">② text prefix</text>
  <text x="331" y="76" text-anchor="middle" fill="#6B675F" font-size="10">"Task: …, State: ⟨discretized state⟩;\nAction: "</text>
  <rect x="502" y="36" width="250" height="56" rx="8" fill="rgba(88,166,255,.13)" stroke="#58a6ff"/>
  <text x="627" y="58" text-anchor="middle" fill="#58a6ff" font-size="12" font-weight="700">③ action tokens + "|"</text>
  <text x="627" y="76" text-anchor="middle" fill="#6B675F" font-size="10">FAST discrete tokens (to be predicted)</text>
  <line x1="10" y1="104" x2="496" y2="104" stroke="#3fb950" stroke-width="2"/>
  <text x="253" y="122" text-anchor="middle" fill="#3fb950" font-size="12">prefix · bidirectional attention · no loss</text>
  <line x1="502" y1="104" x2="752" y2="104" stroke="#58a6ff" stroke-width="2"/>
  <text x="627" y="122" text-anchor="middle" fill="#58a6ff" font-size="12">suffix · causal · loss computed</text>
  <text x="10" y="158" fill="#6B675F" font-size="12">embedding source:</text>
  <text x="85" y="178" text-anchor="middle" fill="#bc8cff" font-size="11">SigLIP vision encoder</text>
  <text x="430" y="178" text-anchor="middle" fill="#bc8cff" font-size="11">Gemma vocabulary embedding (text + action share one table)</text>
  <text x="10" y="208" fill="#6B675F" font-size="11">Note: the state is not a separate vector; it is discretized into 256 buckets and written into the text prompt (as a string).</text>
</svg>
<div class="ln-fig-cap">Figure 9 · The π₀-FAST input sequence structure: images first, then the text prefix, then the action. Green blocks = prefix (image tokens + text prompt/state, bidirectional attention, no loss), blue blocks = postfix (action tokens + "|", causal attention, loss computed); purple annotations mark where each segment's embedding comes from. Takeaway: prefix bidirectional, postfix causal, with text and action sharing one Gemma vocabulary embedding table.</div>
</div>

- **Images first, text second**: each image is passed through SigLIP to obtain a run of image tokens, and **then** the embeddings of the tokenized text prompt are appended.
- **The state lives in the text**: the proprioceptive state is discretized into 256 buckets, turned into a string and spliced into `"State: …;"`, so it travels through the **text embedding** rather than a separate channel.
- **Images share the same Transformer with text/action**, but image tokens attend to each other bidirectionally.

### 16.3 Output: one token predicted per step, autoregressively until the end

The model's output is a **probability distribution over the vocabulary (257,152 classes)**, i.e. "who the next token is". Generation is **autoregressive, token by token**:

```text
while EOS not reached and steps < limit(256):
    token = sample from last_logit          # each step produces exactly [1] token (argmax or temperature sampling)
    embed the token back in, predict the next  # append to the input and continue
stop on EOS or when the limit is reached
```

!!! note "One at a time or several?"
    **One action token per prediction** (one per step). The **total number** of action tokens is **variable** (after FAST compression it is usually far fewer than T×D), terminated by hitting `"|"` / EOS. Training is different: it is **a single forward pass with teacher forcing** — the whole sequence predicts the next token at every position in parallel, and cross-entropy is computed only over the action segment.

### 16.4 Is the decoded result (T, D)? — yes

The generated run of variable-length action tokens is restored: take the tokens between `"Action: "` and `"|"` → map them back into the FAST token space → decode with `time_horizon=T, action_dim=D` and reshape into **`(T, D)`**.

!!! tip "Closing the loop"
    However many tokens sit in the middle, the decoder uses `time_horizon` / `action_dim` to force a reshape back to `(T, D)`, matching exactly the **input** shape of the tokenizer in §1 — and the whole "continuous (T,D) → token → continuous (T,D)" loop closes.

---

<div class="ln-eyebrow">Quick reference · glossary</div>

## Glossary (quick reference)

| Term | Meaning |
|---|---|
| action chunk / horizon (T) | the number of future action steps predicted at once |
| action_dim (D) | the degrees of freedom of each action step |
| tokenization | continuous/text → integer IDs from a finite vocabulary |
| DCT / IDCT | discrete cosine (inverse) transform, time↔frequency, invertible, direct |
| quantization | mapping continuous values onto finite grid points (rounding), the only lossy step |
| BPE | byte pair encoding, lossless compression by merging frequent pairs |
| codebook / codeword | the "dictionary" of a discrete representation and its entries |
| VQ-VAE | a discrete autoencoder with a learnable codebook |
| FSQ | finite scalar quantization, a fixed grid in place of a learnable codebook |
| straight-through estimator | the gradient trick of discrete forward, identity backward |
| mode averaging | regression averaging several correct solutions into a wrong one |
| categorical distribution | a probability distribution over discrete classes, can be multimodal |
| cross-entropy | the negative log-likelihood loss of classification |
| temperature τ | the sampling temperature, tuning the sharpness/diversity of a distribution |
| ar_mask | the mask separating bidirectional (prefix) from causal (action) attention |
| prefix-LM | a mixed Transformer with a bidirectional prefix and a causal suffix |
| flow matching | the continuous action generation route (π₀), of the diffusion family |

---

## Further reading

- [FAST Tokenization paper snapshot](/paper-snapshots/fast-tokenization/) — a close reading of the original FAST paper on paper-snapshots
- [The Fourier transform and DCT](../math/fourier-dct.md) — §4's DCT in full: energy compaction, why it is near-optimal (KLT), and the shared recipe running from JPEG to FAST
- [Autoregressive models · BERT / GPT](../ml/autoregressive-models.md) — the expansion of §11/§13: the three paradigms, BERT bidirectional vs GPT causal, and why VLAs choose prefix-LM

---

## References { #refs }

1. K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, S. Levine. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. arXiv:2501.09747, 2025.
2. K. Black et al. *π₀: A Vision-Language-Action Flow Model for General Robot Control*. RSS 2025. arXiv:2410.24164.
3. R. Sennrich, B. Haddow, A. Birch. *Neural Machine Translation of Rare Words with Subword Units*. ACL 2016. arXiv:1508.07909.
4. N. Ahmed, T. Natarajan, K. R. Rao. *Discrete Cosine Transform*. IEEE Transactions on Computers, 1974.
5. A. Gersho, R. M. Gray. *Vector Quantization and Signal Compression*. Kluwer Academic Publishers, 1992.
6. L. Beyer et al. *PaliGemma: A versatile 3B VLM for transfer*. arXiv:2407.07726, 2024.
7. A. van den Oord, O. Vinyals, K. Kavukcuoglu. *Neural Discrete Representation Learning*. NeurIPS 2017. arXiv:1711.00937.
8. Y. Bengio, N. Léonard, A. Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*. arXiv:1308.3432, 2013.
9. F. Mentzer, D. Minnen, E. Agustsson, M. Tschannen. *Finite Scalar Quantization: VQ-VAE Made Simple*. ICLR 2024. arXiv:2309.15505.
10. A. Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. arXiv:2307.15818, 2023.
11. M. J. Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246, 2024.
12. Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. *Flow Matching for Generative Modeling*. ICLR 2023. arXiv:2210.02747.
13. X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer. *Sigmoid Loss for Language Image Pre-Training*. ICCV 2023. arXiv:2303.15343.
