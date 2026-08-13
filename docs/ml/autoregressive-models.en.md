# Autoregressive Models: BERT / GPT and Prefix-LM Architectures Explained

<div class="ln-byline">2026-08-06 · about 10 min read · Yufeng Jin</div>

<p class="ln-lead" markdown>Which sequence-modelling paradigms exist, what exactly separates BERT from GPT, and why robot VLAs (such as π₀-FAST) use a hybrid of the two called Prefix-LM. This page unfolds the "autoregressive + cross-entropy" and "bidirectional prefix / causal action mask" remarks made in [Action Tokenization](../robotics/action-tokenization.md).</p>

!!! note "One sentence to remember first"
    Every difference reduces to **one thing**: **which other tokens each token is allowed to "see"** (the attention mask). **Bidirectional** sees everything → good at understanding (BERT); **causal** sees only the left → capable of autoregressive generation (GPT). The architecture is the same Transformer block throughout; only the mask and the training objective change.

---

<div class="ln-eyebrow">Preliminaries</div>

## Preliminaries

The main text uses the following concepts over and over. Each is given as "**English term = 中文名 = minimal definition**"; the Chinese name is kept alongside so that English readers can trace the term back into the Chinese literature:

- **self-attention = 自注意力** = the mechanism by which every token in a sequence uses its own query to match the keys of the other tokens and aggregates their values by the resulting weights, thereby extracting information from context<sup>[[1]](#refs)</sup>.
- **attention mask = 注意力掩码** = the boolean matrix specifying "which key positions each query position may attend to"; the difference between the three paradigms of this page (bidirectional / causal / Prefix-LM) lives entirely in this matrix<sup>[[1]](#refs)</sup>.
- **cross-attention = 交叉注意力** = the form of attention in which decoder-side queries attend to the keys/values produced by the encoder, injecting the input condition into the generating side<sup>[[1]](#refs)</sup>.
- **autoregressive = 自回归** = the modelling style that factorises the joint probability into per-token conditionals by the chain rule and predicts the next token from the already-generated history<sup>[[2]](#refs)</sup>.
- **masked language modeling (MLM) = 掩码语言建模** = the training objective that hides part of the input tokens and reconstructs them from the unmasked left and right context<sup>[[3]](#refs)</sup>.
- **cross-entropy = 交叉熵** = the loss function measuring the discrepancy between the predicted and the target distribution; next-token training is exactly the negative log-likelihood of the correct token<sup>[[2]](#refs)</sup>.

---

<div class="ln-eyebrow">Framing 01 · what is estimated</div>

## 0. What is a sequence model actually estimating

Given a string of tokens $x_1,x_2,\dots,x_n$, a language model is essentially estimating their **joint probability** $p(x_1,\dots,x_n)$. How you factorise that joint probability determines the paradigm:

| Paradigm | Factorisation / training objective | Can it generate? |
|---|---|---|
| **Autoregressive (AR)** | $p(x)=\prod_t p(x_t\mid x_{<t})$, predict the next token | ✅ Built for generation |
| **Autoencoding / masked (AE/MLM)** | Hide a portion $\tilde x$, reconstruct $p(x_{\text{masked}}\mid \tilde x)$ | ❌ Mainly for understanding/representation |
| **Sequence-to-sequence (seq2seq)** | Encode the input, decode the output $p(y\mid x)$ | ✅ Translation/summarisation |

!!! tip "Where the name “autoregressive” comes from"
    auto-regressive = regress on your own **already-generated** history $x_{<t}$ to predict the **next** $x_t$, then feed it back into the input and repeat. It is the very same thing as "sample token by token, treat `|` as the terminator" in action generation.

---

<div class="ln-eyebrow">Framing 02 · three stackings</div>

## 1. The three architectural paradigms (one set of bricks, three ways to stack them)

The Transformer brick is "multi-head self-attention + feed-forward layer"<sup>[[1]](#refs)</sup>. The three paradigms differ only in **how you stack them and which mask you use**:

<div class="ln-fig">
<svg viewBox="0 0 820 250" role="img" aria-label="Comparison of three architectures">
  <g>
    <text x="120" y="24" text-anchor="middle" fill="#3fb950" font-size="15" font-weight="700">Encoder-only (BERT)</text>
    <rect x="30" y="40" width="180" height="150" rx="10" fill="#FFFFFF" stroke="#3fb950"/>
    <rect x="55" y="60" width="130" height="34" rx="6" fill="rgba(63,185,80,.15)" stroke="#3fb950"/><text x="120" y="82" text-anchor="middle" fill="#201F1C" font-size="12">Bidirectional attn ×N</text>
    <rect x="55" y="104" width="130" height="26" rx="6" fill="#F1EFEA" stroke="#D8D2C7"/><text x="120" y="121" text-anchor="middle" fill="#6B675F" font-size="11">Feed-forward</text>
    <text x="120" y="160" text-anchor="middle" fill="#6B675F" font-size="11">All inputs visible</text>
    <text x="120" y="178" text-anchor="middle" fill="#6B675F" font-size="11">↑ Per-position representations</text>
  </g>
  <g>
    <text x="410" y="24" text-anchor="middle" fill="#58a6ff" font-size="15" font-weight="700">Decoder-only (GPT)</text>
    <rect x="320" y="40" width="180" height="150" rx="10" fill="#FFFFFF" stroke="#58a6ff"/>
    <rect x="345" y="60" width="130" height="34" rx="6" fill="rgba(88,166,255,.15)" stroke="#58a6ff"/><text x="410" y="82" text-anchor="middle" fill="#201F1C" font-size="12">Causal attn ×N</text>
    <rect x="345" y="104" width="130" height="26" rx="6" fill="#F1EFEA" stroke="#D8D2C7"/><text x="410" y="121" text-anchor="middle" fill="#6B675F" font-size="11">Feed-forward</text>
    <text x="410" y="160" text-anchor="middle" fill="#6B675F" font-size="11">Left context only</text>
    <text x="410" y="178" text-anchor="middle" fill="#6B675F" font-size="11">↑ Predicts the next token</text>
  </g>
  <g>
    <text x="700" y="24" text-anchor="middle" fill="#bc8cff" font-size="15" font-weight="700">Encoder-Decoder (T5)</text>
    <rect x="600" y="40" width="92" height="150" rx="10" fill="#FFFFFF" stroke="#3fb950"/>
    <text x="646" y="115" text-anchor="middle" fill="#3fb950" font-size="12" transform="rotate(-90 646 115)">Encoder (bidir.)</text>
    <rect x="708" y="40" width="92" height="150" rx="10" fill="#FFFFFF" stroke="#58a6ff"/>
    <text x="754" y="115" text-anchor="middle" fill="#58a6ff" font-size="12" transform="rotate(-90 754 115)">Decoder (causal)</text>
    <line x1="692" y1="115" x2="708" y2="115" stroke="#bc8cff" stroke-width="2" marker-end="url(#ar)"/>
    <text x="700" y="208" text-anchor="middle" fill="#6B675F" font-size="11">Linked by cross-attention</text>
  </g>
  <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#bc8cff"/></marker></defs>
</svg>
<div class="ln-fig-cap">Figure 1 · The three architectural paradigms side by side. The same Transformer block; the difference is only "which direction the mask runs" and "how the blocks are stacked". Just read off the type of the attention layer in each box: green = bidirectional attention (encoder side), blue = causal attention (decoder side), purple arrow = the cross-attention that links the two. Takeaway: the paradigms differ not in the bricks but in the mask and the stacking.</div>
</div>

---

<div class="ln-eyebrow">Mechanism · attention mask</div>

## 2. The core mechanism: the attention mask (flip it yourself and see) { #2 }

Self-attention lets every token "query" the other tokens. The **mask** decides which queries are permitted. This is the only essential difference between BERT, GPT and Prefix-LM. In the figure below: row = the token being computed (query), column = the token being attended to (key), **a lit cell = attention allowed**.

<div class="ln-demo">
<div class="ln-demo-title">Figure 2 · Demo: the three attention masks compared</div>
<div class="ln-demo-hint">Click the buttons to switch. The sequence is assumed to be <code>[CLS] task prompt ; A1 A2 A3 |</code>, the first 4 (including the separator <code>;</code>) being the prompt prefix and the last 4 the actions to be generated. What to look at: row = query (who is looking), column = key (being attended to); token labels in green = prompt prefix, in blue = action suffix; lit cell = attention allowed (in Prefix-LM mode the bidirectional region inside the prefix is drawn in green), grey cell = forbidden by the mask. Takeaway: causal = lower triangle, bidirectional = fully lit, Prefix-LM = a bidirectional block at the upper left + a causal triangle at the lower right.</div>
<canvas id="cvMask" width="460" height="460" style="margin:auto"></canvas>
<div class="ln-controls" style="justify-content:center">
<button id="mCausal" class="on">Causal (GPT)</button>
<button id="mBidir">Bidirectional (BERT)</button>
<button id="mPrefix">Prefix-LM (π₀-FAST)</button>
</div>
<div class="ln-readout" id="maskOut"></div>
</div>

- **Causal mask (lower triangle)**: token $t$ may attend only to positions $\le t$. This guarantees "no peeking at the future while predicting the next token", and is the **precondition for autoregressive generation**. GPT uses it throughout.
- **Bidirectional mask (fully lit)**: every token sees the whole sentence. Understanding tasks (classification, extraction) need global context, but **you cannot generate token by token directly** (you would peek at the answer). BERT uses it.

---

<div class="ln-eyebrow">Architecture 01 · BERT</div>

## 3. BERT — encoder / bidirectional / masked language model

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers<sup>[[3]](#refs)</sup>. It uses only the **encoder** stack of the Transformer, with bidirectional attention throughout.

Training objective: **MLM (Masked Language Modeling, a fill-in-the-blanks exercise)**

```text
Input:   机器人 [MASK] 可爱      ← randomly mask out 15% of the tokens
Target:  predict [MASK] = "很"    ← guess it from the context on both sides
```

Because you have to "guess the middle from both sides", the attention must be bidirectional. Another classic objective is NSP (deciding whether two sentences are adjacent)<sup>[[3]](#refs)</sup>, which later work (RoBERTa) found could be dropped<sup>[[4]](#refs)</sup>.

!!! warning "BERT cannot generate directly"
    It excels at "reading" a whole sentence into a vector representation (good for classification, retrieval, named-entity recognition), but because it is bidirectional it **cannot generate autoregressively token by token** — which is why chat and writing tasks go through the GPT branch instead.

| Property | BERT |
|---|---|
| Structure | Encoder-only |
| Attention | Bidirectional |
| Training objective | MLM (+NSP) |
| Strength | Understanding / representation (classification, extraction, retrieval) |
| Can it generate | ❌ |
| Notable descendants | RoBERTa, ALBERT, DeBERTa, ELECTRA |

---

<div class="ln-eyebrow">Architecture 02 · GPT</div>

## 4. GPT — decoder / causal / autoregressive

**G**enerative **P**re-trained **T**ransformer<sup>[[5]](#refs)</sup>. It uses only the **decoder** stack of the Transformer (with the cross-attention to an encoder removed), under a causal mask throughout.

Training objective: **next-token prediction**

$$
\mathcal{L}=-\sum_t \log p_\theta(x_t\mid x_{<t})
$$

This is precisely the objective by which π₀-FAST treats action tokens as language and applies next-token cross-entropy. The demo below shows how autoregression "spits the sequence out one token at a time":

<div class="ln-demo">
<div class="ln-demo-title">Figure 3 · Demo: autoregressive generation (token-by-token sampling)</div>
<div class="ln-demo-hint">Click "Generate next token": the model produces a probability distribution over the next token from the history generated so far, samples from it, and feeds the sample back into the input — this is the loop of GPT and of token-by-token action sampling. What to look at: the boxes along the top = the generated history (each new token is appended on the right), the purple bar chart below = the probability distribution over the next token (each bar is labelled with its probability); drag the temperature slider and you can see the distribution flatten or sharpen, making the sampling correspondingly more random or more deterministic. Takeaway: generation = the loop "predict a distribution → sample → feed back into the input".</div>
<canvas id="cvGen" width="900" height="220"></canvas>
<div class="ln-controls">
<button id="genStep">▶ Generate next token</button>
<button id="genReset">↺ Reset</button>
<label>temperature <input type="range" id="genT" min="10" max="150" value="80"><span class="ln-val" id="genTV">0.80</span></label>
</div>
<div class="ln-readout" id="genOut"></div>
</div>

| Property | GPT |
|---|---|
| Structure | Decoder-only |
| Attention | Causal (unidirectional) |
| Training objective | next-token prediction |
| Strength | Generation (dialogue, writing, code) |
| Can it generate | ✅ (by construction) |
| Notable family | GPT-2/3/4<sup>[[6]](#refs)[[7]](#refs)</sup>, LLaMA, Mistral, Qwen, Gemma, Claude |

---

<div class="ln-eyebrow">Architecture 03 · T5 / BART</div>

## 5. T5 / BART — encoder-decoder (seq2seq)

Put the two branches together: the **encoder** reads the input bidirectionally (say, an English sentence), the **decoder** generates the output causally (say, a Chinese sentence), and **cross-attention** in between lets the decoder read the encoder's representations. This is naturally suited to "input → output" conversion tasks (translation, summarisation).

!!! tip "T5's unifying view"
    T5 turns **every task into "text → text"**: classification = generate the label word, translation = generate the translated text<sup>[[8]](#refs)</sup>. BART instead pre-trains with a denoising objective of "corrupt the text, then reconstruct it"<sup>[[9]](#refs)</sup>. Both are encoder-decoder.

---

<div class="ln-eyebrow">Architecture 04 · Prefix-LM</div>

## 6. Prefix-LM — bidirectional prefix + causal suffix (exactly what π₀-FAST uses)

This is the hinge connecting this page to action tokenization. Prefix-LM (the prefix language model) is a variant of decoder-only<sup>[[8]](#refs)[[10]](#refs)</sup>: split the sequence into a **prefix** and a **suffix**, and use **a single hybrid mask**:

- **Prefix** (prompt + images + discretised state): attends **bidirectionally** within itself (like an encoder, so the condition is fully understood).
- **Suffix** (the action tokens to be generated): attends **causally** (like a decoder, preserving autoregression). The suffix can see the entire prefix.

The "Prefix-LM" button in the demo above (Figure 2) shows you the shape of this mask directly: a fully lit square at the upper left (bidirectional prefix) and a lower triangle at the lower right (causal suffix).

!!! note "Why robot VLAs choose Prefix-LM"
    The observations (images/instructions/state) are **given conditions** and deserve to be understood fully and bidirectionally; the actions are **to be generated** and must therefore be causal. A single Prefix-LM mask satisfies both sides at once — PaliGemma<sup>[[11]](#refs)</sup> and π₀-FAST<sup>[[12]](#refs)</sup> both use it.

---

<div class="ln-eyebrow">Quick reference · family table</div>

## 7. The full model-family table (quick reference)

| Model | Paradigm | Attention | Training objective | Typical use |
|---|---|---|---|---|
| BERT / RoBERTa | Encoder-only | Bidirectional | MLM | Understanding, classification, retrieval |
| GPT / LLaMA / Gemma / Claude | Decoder-only | Causal | next-token | Generation, dialogue |
| T5 / BART | Encoder-Decoder | Bidirectional encoding + causal decoding | Denoising / span reconstruction | Translation, summarisation |
| PaLM / UL2 / PaliGemma | Prefix-LM / hybrid | Bidirectional prefix + causal suffix | (prefix) next-token | Multimodal, conditional generation |
| π₀-FAST | Prefix-LM (VLA) | Bidirectional prefix + causal actions | Cross-entropy on action tokens | Robot action generation |

!!! warning "Why decoder-only is the mainstream today"
    The GPT route (decoder-only + causal) learns representations and gains generation from a single objective, which makes it the simplest to scale — so nearly all large models take this branch. Encoder-only models like BERT remain strong in settings that need understanding only and no generation (retrieval, ranking).

---

<div class="ln-eyebrow">Wrap-up · back to tokenization</div>

## 8. Back to action tokenization

Threading it together (see [Action Tokenization](../robotics/action-tokenization.md) for details):

1. Actions are turned into discrete tokens by a tokenizer (FAST/FSQ/Binning).
2. π₀-FAST uses **Prefix-LM**: the prefix (observations + instruction + state) is bidirectional, the suffix (actions) is causal<sup>[[12]](#refs)</sup>.
3. Training uses **GPT-style next-token cross-entropy**, with the loss computed only over the action segment<sup>[[12]](#refs)</sup>.
4. Inference uses **autoregressive sampling** (the demo on this page, Figure 3), relying on temperature to draw out the different modes of a multimodal solution set.

<div class="ln-lesson" markdown>
**You should now be able to answer**: Why not a pure BERT for a VLA? (It cannot generate.) Why not a purely causal GPT instead of Prefix-LM? (So that the observation condition is understood fully and bidirectionally.) Why does training look like GPT? (next-token cross-entropy + multimodality for free.)
</div>

---

## Further reading

- [Action Tokenization](../robotics/action-tokenization.md) — the main thread on discretising robot actions (FAST / FSQ / Binning + autoregression + multimodality)
- [Fourier Transform and the DCT](../math/fourier-dct.md) — frequency-domain transforms and energy compaction, the mathematical core of FAST's first step

---

## References { #refs }

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. *Attention Is All You Need*. NeurIPS 2017. arXiv:1706.03762.
2. Goodfellow, I., Bengio, Y., Courville, A. *Deep Learning*. MIT Press, 2016.
3. Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019. arXiv:1810.04805.
4. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., Stoyanov, V. *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv:1907.11692.
5. Radford, A., Narasimhan, K., Salimans, T., Sutskever, I. *Improving Language Understanding by Generative Pre-Training*. OpenAI technical report, 2018.
6. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. *Language Models are Unsupervised Multitask Learners*. OpenAI technical report, 2019.
7. Brown, T. B., Mann, B., Ryder, N., et al. *Language Models are Few-Shot Learners*. NeurIPS 2020. arXiv:2005.14165.
8. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P. J. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR 21, 2020. arXiv:1910.10683.
9. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., Zettlemoyer, L. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. ACL 2020. arXiv:1910.13461.
10. Tay, Y., Dehghani, M., Tran, V. Q., et al. *UL2: Unifying Language Learning Paradigms*. arXiv:2205.05131.
11. Beyer, L., Steiner, A., et al. *PaliGemma: A versatile 3B VLM for transfer*. arXiv:2407.07726.
12. Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., Levine, S. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. arXiv:2501.09747.
