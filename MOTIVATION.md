# Why This Project — Mean-Field Intuition, the Readout Cheat, and Reverse-Engineering Circuits

*A heuristic companion to* `[LOW_RANK_REDUCTION.md](LOW_RANK_REDUCTION.md)`
*(which has the methods and numbers). This note explains **why** the project is
set up the way it is. It assumes you know basic 2-layer mean-field theory
(Mei–Montanari–Nguyen / Chizat–Bach style) but nothing about attempts to
extend it to depth.*

Throughout, we distinguish the **intended interpretation** (the working
hypothesis the experiments were designed around) from **speculation** (marked
as such inline). Very little here is proven.

---



## 0. TL;DR

Deep networks don't have a settled mean-field theory the way 2-layer networks
do. Rather than solve that theory problem, we **engineer** the structural
conditions of the mean-field regime into a finite deep network by hand — most
importantly by clamping every readout weight into an `O(1/N)` box. This is a
cheat: it skips the principled routes (scaling limits, Boltzmann sampling) and
works with plain SGD or Adam. The payoff is that the trained solutions come out
*clean*: the network compresses a hierarchical parity task into a **~50-dim
subspace whose size and structure are invariant to width and optimizer** — a
degree-stratified "staircase" that shows up identically in three independent
probes. Solutions that are reproducible objects, rather than optimizer
accidents, are solutions one can realistically hope to **reverse-engineer as
circuits**. That is the long game.

---



## 1. The task: a staircase of parities as order parameters

The target is the **binary-tree staircase** on ±1 inputs: all parities
(monomials) on a complete dyadic tree over 16 bits —

```
degree 2 :  x1x2, x3x4, …, x15x16     (8 pair products)
degree 4 :  x1x2x3x4, …                (4 quartics)
degree 8 :  x1⋯x8, x9⋯x16              (2 octics)
degree 16:  x1x2⋯x16                    (1 full product)
```

— emitted as **15 separate outputs**, with the 16 relevant bits embedded in a
32-bit input (the other 16 bits are noise the model must learn to ignore).

Why this task, from a physics point of view:

- **Parities are an orthogonal basis** (Walsh functions) under the uniform
measure on the hypercube. Each degree is therefore a separately measurable
quantity — we track per-degree test MSE `d2 / d4 / d8 / d16` and treat each
rung as an **order parameter**. When a probe (rank truncation, temperature,
…) is dialed, we don't get one blurry loss curve; we get four rungs turning
on and off independently.
- **Kernels can't do it.** A fixed kernel sees a degree-`k` parity on `d` bits
as one isolated Fourier mode among ~`d^k`, so its sample complexity explodes
with degree. Feature learning is *forced* — the network must discover the
tree structure and climb it degree by degree (this is the "staircase
property" of Abbe et al., which the task is named after).
- **The noise bits force variable selection**, the most elementary kind of
feature learning, and give a built-in null: a model that hasn't learned
anything real correlates with the noise half of the input.



## 2. Recap: what makes 2-layer mean-field theory tick

In the 2-layer mean-field setup you know, the network is

```
f(x) = Σᵢ₌₁ᴺ  aᵢ φ(wᵢ·x),        aᵢ = O(1/N).
```

The whole theory flows from that single scaling choice. Each neuron's
contribution to the output is **intensive** — `O(1/N)` — so the output is an
*average* over the neuron population, and only the **empirical distribution**
of neurons matters:

> No neuron is special. The state of the network is a probability measure
> ρ(a, w) over single-neuron parameters; training is a flow on measures;
> permuting neurons changes nothing.

Contrast the **lazy/NTK scale** `aᵢ = O(1/√N)`: there the output is a
sum of `N` terms of size `1/√N`, which is only finite because of
cancellations — the signal is carried by *fluctuations* around initialization,
individual neurons barely move, and the model linearizes into a kernel
machine. Same architecture, different readout scale, completely different
physics.

The heuristic to hold onto:

> **The readout scale is the dial.** `1/√N` ⇒ fluctuation-dominated, lazy,
> kernel. `1/N` ⇒ average-dominated, collective, feature-learning. And
> `N · (1/N) = O(1)`: an extensive number of intensive contributions gives an
> order-one output — the standard mean-field bookkeeping.



## 3. Depth, and the cheap cheat

For deep networks there is no canonical analogue of the above. The clean
2-layer statement — "the hidden layer is an iid population, the state is a
single measure" — breaks because successive layers are *correlated with each
other* through training; nobody agrees on what the right collective variable
is. We deliberately do **not** try to solve that theory problem here. Instead
we ask an experimentalist's question:

> Can we impose the *structural conditions* of the mean-field regime on a
> finite deep network by hand, cheaply, and check whether the solutions then
> behave as if a mean-field description existed?

The recipe has three ingredients (see `LOW_RANK_REDUCTION.md` §2 for exact
values):

**(i) Frozen orthonormal input projection.** A non-trainable
`Linear(32, N)` with orthonormal columns (`WᵀW = I`, exactly norm-preserving)
lifts the input into `R^N` once and for all. Without it, the trainable
embedding layer "does its own thing" (its norm just sits wherever it likes,
untouched by the regularization) and confounds everything downstream. With it,
*every* trainable layer is a square `N×N` map acting on a common space.

**(ii) Skip connections on every square.** Each layer computes
`h → h + φ(Wh)`: a perturbation added to a shared **residual stream**, rather
than a wholesale re-representation. Depth then means "compose more small
updates on one canvas," which is both closer in spirit to a field-theoretic
picture (one stream, many weak couplings) and what makes the layer-by-layer
probes of §4 well-posed.

**(iii) The readout barrier — the cheat itself.** An L2 hinge on the last
layer,

```
barrier = λ · Σᵢⱼ max(|wᵢⱼ| − c, 0)²,      c ≈ 7/N,  λ = 10,
```

exactly zero inside the per-element box `[−c, c]` and quadratic outside. This
clamps **every readout weight to the mean-field scale** `O(1/N)` (with a
constant ~7, so the total readout mass `N·c` stays `O(1)` — the mean-field
bookkeeping of §2, imposed elementwise).

Why this is a *cheat*: the principled routes to the mean-field regime are
(a) the `α = N` target-rescaling / scaling-limit route, taken carefully as
`N → ∞`, or (b) actually sampling the Boltzmann measure
`e^{−β·MSE − prior}` with Langevin dynamics under the right prior — which is
the parent project's main program, and is slow. The barrier is a **finite-N
shortcut compatible with plain SGD/Adam**: you don't change the
parameterization, you don't take a limit, you don't sample — you just make the
non-mean-field region of weight space energetically inaccessible and let a
cheap optimizer do whatever it wants inside the box.

It matters that the box is *below* where training wants to sit: unconstrained,
the readout spontaneously settles at RMS ≈ `1/√N` — the **lazy scale** (at
`N = 1024`: measured 0.028 vs `1/√N` ≈ 0.031). The barrier at `c ≈ 7/N`
forces it under the lazy/mean-field crossover. So the constraint is genuinely
binding, not decorative.

**A side effect worth savoring: the "humility constraint."** In the data-poor
regime, the unconstrained model's degree-16 output reaches test MSE **1.28** —
*worse than predicting zero*. It is confidently wrong: it amplifies spurious
features with large readout weights. With the barrier, the same budget gives
**0.18**. The intended interpretation: a readout that is elementwise `O(1/N)`
*physically cannot* produce a large output from any single feature, so when
the data carries no real signal the model defaults toward zero instead of
hallucinating. Overconfidence requires an extensive readout weight somewhere;
we made those illegal.

**How do we know the cheat landed in a real regime?** The test we lean on is
**invariance**. Train the recipe at SGD `w=256`, Adam `w=1024`, Adam `w=2048`
(a 4× parameter range *and* an optimizer change): the layer spectra, the
rank-reduction staircase of §4, its per-degree thresholds, and the final
per-degree test MSEs all come out essentially identical. If the solution were
an optimizer accident, none of that would survive. (Speculative, but this is
the working hypothesis: the `c ∝ 1/N` scaling of the box is what makes the
*solution* width-invariant — the constraint strength per neuron is held fixed
as `N` grows, in the same spirit as μ-P hyperparameter transfer.)

## 4. Rank reduction: measuring the circuit's dimensionality

Here is the connection to mean-field thinking. If the trained solution really
is mean-field-like, its physical content is not "a particular `N×N` matrix per
layer" — it is a *distribution* over neuron behaviors, equivalently a
**low-dimensional object embedded in** `R^N`. The width `N` should be mere
ambient space. That is a falsifiable claim:

> PCA-truncate each layer's input, in place, to its top-`k` principal
> components and re-measure the task. If the network genuinely uses `N`
> dimensions, small `k` should wreck it.

(Mechanically: collect each layer's input covariance over held-out data and
replace the input by its rank-`k` PCA reconstruction; skip blocks are handled
in a "pair" formalism so the residual stream and the branch are truncated
coherently — see `LOW_RANK_REDUCTION.md` §3.)

It doesn't wreck it. Two findings:

- **The effective rank is tiny and width-independent.** At every layer, 90% of
representational variance lives in ~30–40 dimensions and 99% in ~40–65 —
whether the network is 256, 1024, or 2048 wide. Top eigenvalues grow with
width; the *number* of significant ones does not. The same compressed code
is re-embedded in bigger and bigger ambient spaces.
- **Each degree "cracks" at its own rank** — the staircase again, now in `k`:
degree 2 recovers at `k ≈ 37`, degree 4 at `k ≈ 42`, degree 8 at `k ≈ 46`,
degree 16 at `k ≈ 48–64`. **Each octave of degree costs ~4–5 more principal
components.** Low-degree features are the most compactly encoded; the full
parity is the most expensive and has the softest transition.

The intended interpretation: the dimension of the solution is set by the
**task structure plus the recipe**, not by the architecture or the optimizer.
`N` is headroom the model doesn't use. For a mean-field mindset this is
exactly the expected signature — the "state" is the ~50-dim code, and width
only controls how comfortably it is embedded.

## 5. The circuit picture: the same staircase in the neuron basis

Rank reduction sees the solution in the PCA basis. The second probe asks what
it looks like in the **neuron basis**: correlate every neuron's activation
with every one of the 15 tree monomials, then classify neurons
(single-feature / degree-pure polysemantic / cross-degree mixed). The result
is a compositional hierarchy:

- **Layer 0 is a labeled-line code**: ~83% of its neurons carry exactly one
pair or one quartic each, with no leakage — crisp, axis-aligned, the kind of
thing you could read off with a ruler.
- **Middle layers develop degree-pure mixing**: by layer 3, a population of
neurons each carries *all eight* pair products at magnitude ≈ `1/√8`, with
signed patterns (Walsh–Hadamard-like — *speculative*; we haven't verified
the sign structure is literally a Hadamard code).
- **The pre-readout layer is 100% cross-degree mixed**: no pure neuron
survives to the end; everything has been composed. Dead neurons are <2%
everywhere — the code is compact but not sparse-by-death.

So the network builds parities the way you would: local products first,
recursive composition after, with the representation getting progressively
more mixed as it climbs the tree. This — *which* features exist, *where* they
get composed, in *what* basis — is what we mean by the **circuit**.

And this is the point of the whole exercise. Reverse-engineering circuits in
ordinary trained networks is notoriously slippery: solutions differ across
seeds, optimizers, widths; any structure you find might be an artifact of one
training run. The bet made here (intended interpretation, not a theorem) is:

> In a mean-field-like regime the solution is pinned by the task and the
> prior, not by the optimization path. Circuits then become **identifiable,
> reproducible objects** — the same ~50 dimensions, the same staircase, the
> same hierarchy at every width and under two different optimizers — and
> reverse-engineering them stops being archaeology and starts being physics.

The concrete open program: identify what the ~50 dimensions *are* (they encode
15 outputs plus intermediates — how much superposition?), pin down the
composition map layer by layer, and understand why each octave of degree costs
precisely a few extra dimensions.

## 6. Coda: three probes, one structure

We actually have a third, independent probe: **tempering** — sample around the
trained minimum at increasing temperature and watch the rungs melt. Higher
degrees melt first, degree 2 is the most robust: the same staircase, now in
temperature. Three probes with unrelated failure modes (linear-algebraic
truncation, single-neuron statistics, thermal noise) recovering one structure
is the main reason we believe the structure is real.

Where to go next:

- `[LOW_RANK_REDUCTION.md](LOW_RANK_REDUCTION.md)` — methods, tables, exact
numbers, and the reproduce-it-yourself commands (a trained checkpoint ships
in this repo).
- `[NOTEBOOK.md](NOTEBOOK.md)` — the chronological lab notebook these claims
are distilled from.
- Caveats live in `LOW_RANK_REDUCTION.md` §8 and apply to everything above:
single seed, infinite-data regime, and all results tied to this specific
recipe. The mean-field interpretation is a working hypothesis we are trying
to break, not a result.

