# Fourier Transform & DCT: From Frequency to Energy Compaction

<div class="ln-byline">2026-08-06 · about 11 min read · Yufeng Jin</div>

<p class="ln-lead" markdown>Starting from Fourier series and running all the way to the energy-compaction property of the DCT, this note builds a genuine understanding of why applying a DCT to a smooth signal (a robot motion, say) makes its energy pile up almost miraculously onto a handful of low-frequency coefficients. It is the expanded version of the step that [Action Tokenization](../robotics/action-tokenization.md) treats as a black box.</p>

!!! note "One through-line"
    **Any signal can be written as a superposition of "waves."** What the Fourier family studies is *which* waves to use and *how much* of each. The DCT is the member of that family tuned specifically for **real-valued, smooth signals**, and its killer feature is **energy compaction** — squeezing the information into very few low-frequency coefficients. That is precisely the foundation on which FAST and JPEG compression rest.

---

<div class="ln-eyebrow">Opening · why transform</div>

## 0. Why move a signal into the "frequency domain"

The same stretch of signal admits two views:

- **Time domain**: "what is the value at each instant." A robot motion is natively of this kind: the joint angles at step 0, step 1, and so on.
- **Frequency domain**: "which fast and slow waves make up the signal, and in what proportion." Another coordinate system for the same information.

The payoff of switching to the frequency domain: many signals that look densely packed in time are **extremely sparse** in frequency (only a few frequencies carry any value). Smooth = slowly varying = low frequencies only = sparse in frequency → **compressible**.

---

<div class="ln-eyebrow">Prerequisites</div>

## Prerequisites

The main text leans on the following concepts repeatedly; each is written as an "English term = Chinese name = minimal definition" triple:

- **periodic function** = 周期函数 = a function for which there exists a positive number $T$ with $f(t+T)=f(t)$ for all $t$; $T$ is called the period<sup>[[1]](#refs)</sup>.
- **harmonic** = 谐波 = a sine/cosine component whose frequency is an integer multiple of the fundamental; the $n$-th harmonic has $n$ times the fundamental frequency<sup>[[1]](#refs)</sup>.
- **complex exponential** = 复指数 = a complex-valued function of the form $e^{i\omega t}$; by Euler's formula $e^{i\omega t}=\cos\omega t + i\sin\omega t$, it packs a sine and a cosine into a single complex quantity<sup>[[1]](#refs)</sup>.
- **basis function** = 基函数 = one of a fixed set of functions/vectors used for expansion; a signal is written as their weighted sum, and the weights are the transform coefficients<sup>[[1]](#refs)</sup>.
- **energy compaction** = 能量压缩 = the property that a transform concentrates most of a signal's energy onto a few coefficients; the better the concentration, the more compressible the signal<sup>[[2]](#refs)</sup>.
- **quantization** = 量化 = the lossy operation of mapping continuous or high-precision values onto a finite set of discrete levels (rounding, for instance)<sup>[[2]](#refs)</sup>.

---

<div class="ln-eyebrow">Theory 01 · Fourier series</div>

## 1. Fourier series: a periodic signal = a sum of sines and cosines

Fourier's insight (1807): any **periodic** function can be written as a weighted sum of sines and cosines at different frequencies<sup>[[3]](#refs)</sup>:

$$
f(t)=\frac{a_0}{2}+\sum_{n=1}^{\infty}\Big(a_n\cos(n\omega t)+b_n\sin(n\omega t)\Big)
$$

The higher the frequency $n\omega$, the faster that term varies. Below we **approximate a square wave by superposing odd harmonics** — the classic example, and one in which you can also watch the **Gibbs phenomenon**: the more terms you add the better the fit, yet the corners always overshoot.

<div class="ln-demo">
<div class="ln-demo-title">Figure 1 · Fourier series approximating a square wave</div>
<div class="ln-demo-hint">A square wave = the sum of infinitely many odd sine harmonics. The grey curve is the target square wave, the red curve is the superposition of the first few harmonics; drag the slider to add harmonics and watch the sum (red) close in on the square wave (grey). What to look for: more harmonics means a tighter overall fit, yet the overshoot of the red curve near the jumps never disappears — that is the Gibbs phenomenon discussed in the text.</div>
<canvas id="cvSquare" width="900" height="280"></canvas>
<div class="ln-controls">
<label>Number of harmonics <input type="range" id="sqN" min="1" max="40" value="3"> <span class="ln-val" id="sqNV">3</span></label>
</div>
<div class="ln-readout" id="sqOut"></div>
</div>

!!! warning "The Gibbs phenomenon"
    Near a discontinuity (the jump in a square wave) there is always an overshoot of about 9%, no matter how many terms you add<sup>[[1]](#refs)</sup>. The lesson: **the more discontinuous a signal is, the more high-frequency content it carries, and the harder it is to represent with a few low-frequency terms** — conversely, **the smoother it is, the better it compresses**. Robot motions happen to be smooth.

---

<div class="ln-eyebrow">Theory 02 · continuous transform</div>

## 2. The continuous Fourier transform: from periodic to arbitrary signals

Push the period to infinity and the series' "sum over discrete frequencies" becomes an "integral over continuous frequency," which gives the Fourier transform:

$$
F(\omega)=\int_{-\infty}^{\infty} f(t)\,e^{-i\omega t}\,dt
$$

Here $e^{-i\omega t}=\cos\omega t - i\sin\omega t$ (Euler's formula) packs sine and cosine into a single **complex exponential**. $F(\omega)$ is complex: its **magnitude** gives the strength of that frequency, its **argument** the phase.

!!! tip "A complex exponential is a rotating phasor"
    In the complex plane $e^{-i\omega t}$ is a unit vector rotating at angular frequency $\omega$. The Fourier transform is asking: "how much of the signal is in sync with this phasor spinning at $\omega$?" Frequency components that stay in sync come out large.

---

<div class="ln-eyebrow">Theory 03 · discrete and fast</div>

## 3. DFT and FFT: Fourier in the discrete world

Inside a computer a signal is **a finite set of samples** $x_0,\dots,x_{N-1}$, which calls for the **discrete Fourier transform (DFT)**:

$$
X_k=\sum_{n=0}^{N-1}x_n\,e^{-i2\pi kn/N},\quad k=0,\dots,N-1
$$

Computed directly this costs $O(N^2)$. The **FFT (fast Fourier transform)** exploits symmetries to bring it down to $O(N\log N)$<sup>[[4]](#refs)</sup> — one of the most important algorithms of the 20th century, and what made real-time audio and video processing possible.

| Name | Input | Output | Complexity |
|---|---|---|---|
| Fourier series | continuous periodic function | discrete coefficients $a_n,b_n$ | — |
| Fourier transform (FT) | continuous, non-periodic | continuous spectrum $F(\omega)$ | — |
| DFT | $N$ discrete samples | $N$ complex coefficients | $O(N^2)$ |
| FFT | same as DFT (algorithmic speedup) | same as DFT | $O(N\log N)$ |
| **DCT** | $N$ real samples | $N$ **real** coefficients | $O(N\log N)$ |

---

<div class="ln-eyebrow">DCT 01 · origin</div>

## 4. From DFT to DCT: dropping the complex numbers and the boundary jump

The DFT has two features that are unfriendly to compression, and the DCT exists precisely to fix them:

1. **DFT coefficients are complex** (real and imaginary parts), which is redundant for a real-valued signal.
2. **The DFT implicitly extends the signal periodically**, so if the first and last values differ, an **artificial jump** appears at the boundary → a flood of high frequencies → bad for compression.

!!! note "The DCT's two tricks"
    ① **Even symmetric extension**: the DCT first "mirrors" the signal into an even function before transforming. The Fourier expansion of an even function **contains cosine terms only** → the coefficients are all **real**, with no imaginary-part redundancy.

    ② **Mirroring kills the jump**: the mirrored extension makes the boundary **join up smoothly** (no more DFT-style discontinuity between the ends) → high frequencies are suppressed → energy concentrates further into the low frequencies.

    Together these two points make the DCT's **energy compaction** on real, smooth signals markedly better than the DFT's.

The workhorse is **DCT-II** (the one JPEG and FAST use)<sup>[[5]](#refs)</sup>:

$$
X_k=\sum_{n=0}^{N-1}x_n\cos\!\Big[\frac{\pi}{N}\big(n+\tfrac12\big)k\Big]
$$

Note that the basis is pure cosine and the coefficients $X_k$ are real. $k=0$ is the DC term (the mean), and larger $k$ means higher frequency. The inverse transform, the IDCT (i.e. DCT-III), reconstructs using the same cosine basis and is **exactly invertible**.

---

<div class="ln-eyebrow">DCT 02 · variants</div>

## 5. The four DCT variants (knowing they differ is enough)

Depending on how the symmetric extension is performed at the two ends, there are four types, DCT-I through IV<sup>[[2]](#refs)</sup>. In practice:

| Variant | Use |
|---|---|
| **DCT-II** | The most common. JPEG, MPEG, and **FAST action tokenization** all use it |
| **DCT-III** | The inverse of DCT-II (i.e. the IDCT) |
| DCT-I | Different endpoint handling, rarely used |
| DCT-IV | Used in the MDCT (audio, e.g. the lapped transform in MP3/AAC) |

!!! tip "What norm=\"ortho\" does"
    In scipy one often adds `norm="ortho"`: it multiplies the coefficients by normalization factors ($\sqrt{1/N}$ for $k{=}0$, $\sqrt{2/N}$ for the rest), making the transform **orthogonal and energy-preserving** (Parseval: energy in time = energy in frequency). The IDCT then becomes the transpose of the DCT, which is numerically cleaner.

---

<div class="ln-eyebrow">DCT 03 · basis functions</div>

## 6. A gallery of DCT basis functions

The DCT decomposes a signal onto this set of **fixed cosine basis vectors**. The $k$-th basis vector = a sampled cosine of frequency $k$. Any signal is a weighted sum of these bases, and the weights are the DCT coefficients. Here are the first 8 bases ($N=32$):

<div class="ln-demo">
<div class="ln-demo-title">Figure 2 · DCT-II basis functions (k = 0…7)</div>
<div class="ln-demo-hint">In each panel the horizontal axis is the sample index n and the vertical axis the value of that basis vector: k=0 is constant (DC / mean), and the larger k is the faster the oscillation (the higher the frequency). Hover over any panel to highlight it. What to look for: all the DCT does is project the signal onto this fixed set of cosine waves ordered from slow to fast, and each projection is a coefficient.</div>
<canvas id="cvBasis" width="900" height="320"></canvas>
<div class="ln-readout">Each panel is one cosine basis vector $\cos[\pi/N\,(n+0.5)\,k]$. The DCT coefficient $X_k$ = the projection of the signal onto the k-th basis vector.</div>
</div>

---

<div class="ln-eyebrow">Evidence · energy compaction</div>

## 7. Energy compaction: the DCT's killer feature (verify it yourself) { #7-dct }

This is the single most important concept on the page, and the fundamental reason FAST works. **Energy compaction**: for a smooth signal, the DCT concentrates the overwhelming majority of the energy onto **a very small number of low-frequency coefficients**, leaving the rest close to 0.

So we need only **keep the first few large coefficients and discard the great many that sit close to 0** to reconstruct the signal almost perfectly from very little data. Verify it for yourself below:

<div class="ln-demo">
<div class="ln-demo-title">Figure 3 · Reconstruction quality when keeping only the first k DCT coefficients</div>
<div class="ln-demo-hint">Top: the original signal (blue) vs the reconstruction from only the first k DCT coefficients (red). Bottom: the energy of the DCT coefficients (the retained ones highlighted). Drag k and see how few coefficients a smooth signal needs before the curves nearly coincide; then switch to the "with high frequency / noise" signal for comparison. What to look for: for the smooth signal almost all the energy is crammed into the leftmost few low-frequency coefficients, so red and blue overlap at a very small k, whereas for the signal with high frequency content the coefficient energy is spread out and a far larger k is needed — that is exactly what energy compaction means, made visible.</div>
<canvas id="cvCompact" width="900" height="360"></canvas>
<div class="ln-controls">
<label>Keep the first k coefficients <input type="range" id="ckK" min="1" max="32" value="4"> <span class="ln-val" id="ckKV">4</span> / 32</label>
<button id="ckSmooth" class="on">Smooth signal</button>
<button id="ckRough">With high frequency / noise</button>
</div>
<div class="ln-readout" id="ckOut"></div>
</div>

!!! note "The link to FAST"
    FAST does not explicitly "discard" coefficients; it **rounds them (quantization)**: the high-frequency coefficients that sit close to 0 turn into **exactly 0** once rounded, which has the same effect as discarding them and, as a bonus, produces long runs of repeated 0s for BPE to compress<sup>[[6]](#refs)</sup>. The stronger the energy compaction (the smoother the signal), the more 0s after rounding → the fewer tokens.

---

<div class="ln-eyebrow">Rationale · near-optimal</div>

## 8. Why the DCT of all transforms (it is close to "optimal")

In theory, for a given signal statistic the transform with **optimal energy compaction** is the **KLT (Karhunen–Loève transform, i.e. PCA)** — it projects the signal onto the eigenvectors of the covariance matrix. But the KLT depends on the data statistics and requires computing eigenvectors on the spot: expensive and not general-purpose.

!!! tip "The DCT ≈ a free KLT"
    For a **first-order Markov signal** (adjacent samples highly correlated — exactly the case for smooth signals), one can prove that **the DCT's basis functions come arbitrarily close to the KLT's optimal basis**<sup>[[5]](#refs)</sup><sup>[[2]](#refs)</sup>. In other words: with a **fixed, training-free** cosine basis, the DCT achieves nearly the compression of "the optimal transform tailored to this class of signals." That is why JPEG, MPEG, and FAST all reach for the DCT instead of actually computing a KLT.

<div class="ln-lesson" markdown>
Echoing learned vs direct: the DCT is **direct** (fixed basis, zero training), yet it approaches the optimal learned transform that would require data statistics — **deterministic mathematics buying you nearly the benefits of learning**.
</div>

---

<div class="ln-eyebrow">Applications · JPEG to FAST</div>

## 9. Applications: from JPEG to FAST

| System | How the DCT is used |
|---|---|
| **JPEG** | Cut the image into 8×8 blocks → 2D DCT → quantize (drop the high frequencies) → entropy coding<sup>[[7]](#refs)</sup>. Nearly every photo you have ever seen has been through a DCT |
| **MPEG / H.26x** | DCT + quantization on video frames (residuals) |
| **MP3 / AAC** | Compress audio with the MDCT (the lapped version of DCT-IV) |
| **FAST (robotics)** | DCT of the action array (T,D) along the **time axis** → quantize by rounding → BPE<sup>[[6]](#refs)</sup>. Smooth motions → concentrated energy → many 0s after rounding → short token sequences |

!!! tip "One and the same recipe"
    JPEG compressing images and FAST compressing actions have **exactly the same skeleton**: move to the frequency domain (DCT) → quantize away the unimportant high frequencies → lossless entropy coding (BPE/Huffman). Understand JPEG and you already understand half of FAST.

---

## Further reading

- [Action Tokenization](../robotics/action-tokenization.md) — the main thread on discretizing robot actions; FAST uses the DCT to compress motions into tokens
- [Autoregressive Models · BERT / GPT](../ml/autoregressive-models.md) — sequence-modeling paradigms and the Prefix-LM, i.e. who consumes the tokens afterwards

---

## References { #refs }

1. A. V. Oppenheim, R. W. Schafer. *Discrete-Time Signal Processing*, 3rd ed. Prentice Hall, 2010.
2. K. R. Rao, P. Yip. *Discrete Cosine Transform: Algorithms, Advantages, Applications*. Academic Press, 1990.
3. J. B. J. Fourier. *Théorie analytique de la chaleur*. Firmin Didot, Paris, 1822. (The book-length expansion of the heat-conduction memoir submitted to the Paris Academy of Sciences in 1807.)
4. J. W. Cooley, J. W. Tukey. "An Algorithm for the Machine Calculation of Complex Fourier Series." *Mathematics of Computation*, 19(90): 297–301, 1965.
5. N. Ahmed, T. Natarajan, K. R. Rao. "Discrete Cosine Transform." *IEEE Transactions on Computers*, C-23(1): 90–93, 1974.
6. K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, S. Levine. "FAST: Efficient Action Tokenization for Vision-Language-Action Models." arXiv:2501.09747, 2025.
7. G. K. Wallace. "The JPEG Still Picture Compression Standard." *IEEE Transactions on Consumer Electronics*, 38(1): xviii–xxxiv, 1992.
