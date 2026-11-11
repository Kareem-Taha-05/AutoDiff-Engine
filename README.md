# AutoDiff Engine

> An optimization lab in pure C++17: gradient descent, SGD, momentum and adaptive optimizers, non-convex benchmarks and live Raylib visualizations. No ML libraries, no abstraction layers, full math exposed.

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![Build](https://img.shields.io/badge/build-CMake-green)
![Visualization](https://img.shields.io/badge/visualization-Raylib-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What is this?

A personal engineering project where I implement the foundational pieces of ML optimization **completely from scratch in C++**: a strided tensor data structure, gradient descent in two forms (numerical and analytical), mini-batch SGD, and a set of modern optimizers (Momentum, AdaGrad, RMSprop, Adam) that can be compared head to head on the same data, the same start point and the same batches.

No PyTorch. No Eigen. No NumPy. Just C++, math, and pixels.

I built this to make sure I *genuinely understand* what happens when you call `optimizer.step()`, every multiply, every divide, every gradient accumulation, and then to turn that understanding into an actual experiment you can run and measure.

---

## What it demonstrates

| Concept | Implementation |
|---|---|
| Custom strided Tensor | `include/tensor.hpp`: Rule of 5, `std::vector` backing, bounds-checked `operator()` |
| Mean Euclidean Distance Loss | `src/optimization.cpp::euclidean_loss()` |
| GD: Numerical gradient | Finite-difference: `[L(x+h) − L(x)] / h` |
| GD: Analytical gradient | Closed-form partial derivatives, exact |
| Mini-batch SGD | Fisher-Yates shuffle + batched gradient accumulation |
| Optimizer comparison | `optim::run_optimizer()`: GD, Momentum, AdaGrad, RMSprop, Adam behind one update interface |
| Non-convex benchmarks | `optim::run_analytic_optimizer()`: Rosenbrock (curved valley) and Rastrigin (grid of local minima) with exact gradients |
| Dataset generators | `data::` uniform, Gaussian, clustered, injected outliers, fixed seeds, CSV save/load |
| Loss contour map | `viz::plot_loss_contour()`: banded loss heatmap with every optimizer's trajectory on top |
| Gradient vector field | Hold `G` to draw the descent direction (−∇L) across the whole search space |
| Experiment reports | `report::` console summary, per-optimizer summary CSV, per-epoch loss history CSV |
| Live Raylib plots | Scrollable/zoomable scatter plots and loss curves with legends |

---

## What you'll see when you run it

**Scatter plot window**
Your dataset in black. After optimization converges, the single red dot marks where point `p` ended up, the geometric median of the cloud.

**Loss curve window**
A falling curve per epoch. Full-batch GD (both forms) gives a smooth descent. SGD gives a noisy, jumpier curve, that's the gradient variance from mini-batches. Both converge. That's the point.

**Loss contour window (new)**
The loss landscape drawn as colored bands (blue = low loss, red = high loss) with one trajectory per optimizer and a legend. Hold **`G`** to overlay the gradient field: one arrow per grid spot showing where gradient descent would step next. You can literally watch Momentum overshoot and swing back while plain GD takes the direct route.

Windows support **mouse-wheel zoom** (scatter plots) and stay open until you close them.

---

## Optimizer comparison

Every optimizer is run on **exactly the same** dataset, start point, learning rate and mini-batch sequence (the shuffle is seeded per run), so the only thing that changes is the update rule. Each run records:

- initial and final loss (measured on the *whole* dataset at the end of every epoch, so curves are comparable)
- epochs to convergence (`|ΔL| < tolerance`, reported as *not reached* if it never happens)
- number of parameter updates
- training time
- the full trajectory of `p`

### Update rules

```
GD        p ← p − lr · g
Momentum  m ← β·m + g                      p ← p − lr · m
AdaGrad   v ← v + g²                       p ← p − lr · g / (√v + ε)
RMSprop   v ← β·v + (1−β)·g²               p ← p − lr · g / (√v + ε)
Adam      m ← β₁·m + (1−β₁)·g
          v ← β₂·v + (1−β₂)·g²
          m̂ = m/(1−β₁ᵗ)   v̂ = v/(1−β₂ᵗ)    p ← p − lr · m̂ / (√v̂ + ε)
```

All five share one `optim::apply_step()` function (per-parameter rule in `optim::step_1d()`), so adding a new optimizer means adding one `case`.

### Sample results

Euclidean-distance loss, 100 points in 3 Gaussian clusters plus 5 outliers, batch size 10, lr = 1.0, tolerance = 0.001, start at (1, 1), seed 100:

| Optimizer | Final loss | Updates | Convergence |
|---|---|---|---|
| GD | 19.84 | 154 | reached (epoch 14) |
| Momentum | 19.87 | 132 | reached (epoch 12) |
| AdaGrad | 20.07 | 1100 | not reached |
| RMSprop | 19.92 | 1100 | not reached |
| Adam | 19.84 | 110 | reached (epoch 10) |

AdaGrad and RMSprop keep a step size of roughly `lr` regardless of how small the gradient gets, so with a fixed lr of 1.0 they jitter around the minimum by more than the tolerance. Lowering lr or loosening the tolerance changes that. Results depend on seed and hyperparameters, so treat the table as an example run, not a benchmark.

---

## Non-convex objectives

The optimizers are not tied to the dataset loss. Two classic 2D test functions, with exact analytic gradients, let you see how each optimizer handles hard landscapes:

| Function | Formula | Global minimum | Why it's hard |
|---|---|---|---|
| Rosenbrock | `(1−x)² + 100(y−x²)²` | 0 at (1, 1) | long, narrow, curved valley, steep walls, flat floor |
| Rastrigin | `20 + Σ [xᵢ² − 10·cos(2πxᵢ)]` | 0 at (0, 0) | a regular grid of local minima |

Each optimizer gets its own learning rate in this experiment, since a rate that suits Adam makes plain GD diverge on Rosenbrock. If the loss goes to `inf`/`NaN` the run stops instead of plotting garbage.

Switch landscapes with the `USE_ROSENBROCK` constant at the top of `examples/ex5_nonconvex.cpp`.

Sample results (5000 epochs max, tolerance 1e-6, start at (−1.5, 1.5) for Rosenbrock and (3.5, 3.5) for Rastrigin):

| Optimizer | Rosenbrock final loss | Rastrigin final loss |
|---|---|---|
| GD | 0.00495 (not reached) | 17.91 |
| Momentum | 0.00060 (epoch 3622) | 17.91 |
| AdaGrad | 0.368 (not reached) | 17.91 |
| RMSprop | 0.0833 (epoch 744) | 17.91 |
| Adam | 0.00012 (epoch 1915) | 17.91 |

Two things worth knowing about these numbers. On Rosenbrock, RMSprop's "reached" at 0.083 is the tolerance firing early on the flat valley floor, not a true minimum. On Rastrigin from this start point all five optimizers fall into the same local minimum (loss ≈ 17.9, the global minimum is 0): gradient methods follow the slope they start on, which is exactly what this landscape is there to show. Try other start points or learning rates to see them separate.

---

## Datasets

| Generator | Description |
|---|---|
| Uniform | Points uniformly distributed in `[min, max]²` (the original generator) |
| Gaussian | Points around a center with a chosen standard deviation |
| Clustered | N cluster centers, points scattered around them, clamped to the plot area |
| Outliers | Extra points placed near the corners, far from everything else |

All generators take an explicit **seed** (own `std::mt19937`, the global `rand()` state is not touched), so any experiment can be reproduced exactly. Datasets can be saved to and loaded from CSV.

---

## Experiment reports

Every comparison writes two CSV files (into the directory you run the binary from) that open directly in Excel or pandas:

**`experiment_report.csv`**: one row per optimizer

```
optimizer,initial_loss,final_loss,epochs_to_converge,updates,time_ms,converged
```

**`loss_history.csv`**: one row per (optimizer, epoch), ready to plot with one line per optimizer

```
optimizer,epoch,loss
```

---

## Project structure

```
AutoDiff-Engine/
│
├── include/
│   ├── tensor.hpp          # Strided rank-1/2/3 tensor (custom, no Eigen)
│   ├── point.hpp           # Simple 2D point struct
│   ├── optimization.hpp    # GD / SGD / optimizers, Optimizer + Objective enums, OptimizerResult
│   ├── data_generation.hpp # Uniform / Gaussian / clustered / outliers, CSV I/O, shuffling
│   ├── visualization.hpp   # Raylib scatter, loss curve, contour map, gradient field
│   └── report.hpp          # Console + CSV experiment reports (namespace report)
│
├── src/
│   ├── optimization.cpp    # GD, SGD, Momentum/AdaGrad/RMSprop/Adam, Rosenbrock/Rastrigin
│   ├── data_generation.cpp # Dataset generators, Fisher-Yates shuffle, CSV save/load
│   ├── visualization.cpp   # Raylib render loops, axes, heatmap, trajectories, legends
│   └── report.cpp          # Summary and loss-history CSV writers
│
├── examples/
│   ├── ex1_limit_gd.cpp          # GD with finite-difference (numerical) gradient
│   ├── ex2_closed_gd.cpp         # GD with exact analytical gradient
│   ├── ex3_sgd.cpp               # Mini-batch SGD
│   ├── ex4_optimizer_compare.cpp # All optimizers on one dataset + contour map + reports
│   └── ex5_nonconvex.cpp         # Rosenbrock / Rastrigin with every optimizer
│
└── CMakeLists.txt
```

---

## Build & run

### Prerequisites

```bash
# Ubuntu / Debian
sudo apt install build-essential cmake git libgl1-mesa-dev libx11-dev \
     libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

> Raylib is **fetched and compiled automatically** by CMake if it's not found on your system.

### Build

```bash
git clone https://github.com/Kareem-Taha-05/AutoDiff-Engine
cd AutoDiff-Engine

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Binaries land in `build/bin/`.

### Run

```bash
# Gradient descent: numerical gradient (finite-difference)
./build/bin/ex1_limit_gd

# Gradient descent: exact analytical gradient
./build/bin/ex2_closed_gd

# Mini-batch SGD (batch_size = 10)
./build/bin/ex3_sgd

# Optimizer comparison: prints a report, writes the CSVs, opens the contour map
./build/bin/ex4_optimizer_compare

# Non-convex objectives (Rosenbrock / Rastrigin)
./build/bin/ex5_nonconvex
```

Each binary opens its Raylib windows in sequence. Close a window with `Esc` or the title-bar button to proceed. In the contour window, hold `G` to show the gradient field.

---

## The Tensor

The `Tensor<T>` struct is the backbone of the project. It models a rank-1/2/3 strided array and is built entirely from scratch.

**Key design decisions:**

- **`std::vector<T>` backing store:** instead of raw `new[]`/`delete[]`. This gives us correct copy, move, and destruction semantics for free via the compiler-generated Rule of Five, and makes the constructor exception-safe without any extra code.
- **Explicit stride layout:** `stride_x = length_y * length_z`, `stride_y = length_z`, `stride_z = 1`, the same row-major layout used by NumPy and most linear algebra libraries.
- **Bounds-checked `operator()(i, j, k)`:** active in Debug builds, compiled out in Release via `NDEBUG`.
- The Rule of Five is declared `= default` explicitly to document intent even though `std::vector` makes them redundant.

```cpp
Tensor<float> weights(128, 64);     // 2D: 128 rows × 64 cols
weights(3, 7) = 0.42f;              // operator(), no manual index arithmetic
```

---

## The Math (quick reference)

### Loss
```
L(p) = (1/N) · Σᵢ √[(xᵢ − x)² + (yᵢ − y)²]
```

### Numerical gradient (limit form)
```
∂L/∂x ≈ [L(p + h·eₓ) − L(p)] / h       h = 0.001
```

### Analytical gradient (closed form)
```
∂L/∂x = −(1/N) · Σᵢ (xᵢ − x) / √[(xᵢ − x)² + (yᵢ − y)²]
```

The gradient is undefined when `p` lands exactly on a data point (division by zero), so that point's term is skipped instead of letting `inf`/`NaN` spread through the run.

### Update rule (plain GD)
```
p ← p − lr · ∇L(p)
```
See [Optimizer comparison](#optimizer-comparison) for the Momentum and adaptive variants.

### Rosenbrock gradient
```
∂f/∂x = −2(1−x) − 400x(y−x²)        ∂f/∂y = 200(y−x²)
```

### Rastrigin gradient
```
∂f/∂xᵢ = 2xᵢ + 20π·sin(2πxᵢ)
```

---

## Design notes

- **Fair comparisons by construction.** Dataset and start point are passed by value so a shuffle in one run can't leak into the next, and each run is seeded identically so every optimizer sees the same batches.
- **One update interface.** The per-parameter update rule is a single function; the dataset optimizer and the analytic-function optimizer both call it.
- **Reproducibility.** Every random component takes an explicit seed.
- **Honest metrics.** The per-epoch loss is the full-dataset loss, not the average of noisy batch losses.

---

## Roadmap / TODOs

Done:

- [x] **Momentum SGD**, plus AdaGrad, RMSprop and Adam, compared on identical data
- [x] **Gradient direction field** drawn over the loss contour map
- [x] **Loss contour map** with optimizer trajectories
- [x] **Non-convex objectives**: Rosenbrock and Rastrigin
- [x] **Dataset generators** with seeds, outliers and CSV save/load
- [x] **Experiment reports** as CSV

Next:

- [ ] **3D loss surface:** render an X×Y mesh of loss values as a Raylib 3D surface using `DrawMesh`, so you can visually see the bowl shape and watch the point descend into it.
- [ ] **Live loss graph** with pause / step / reset controls (needs the optimizer loop split into a single-update function)
- [ ] **Nesterov momentum** (needs the gradient at a lookahead position)
- [ ] **MSE / MAE / Huber objectives** (needs a model such as `y = ax + b` to fit, which also enables the noisy-linear dataset)
- [ ] **Tensor reshape / transpose:** implement `reshape(nx, ny)` returning a view with new strides but shared data (no copy), and a 2D `transpose()`.
- [ ] **Lecture 5+ content:** extend to linear regression and a from-scratch single-layer network using the existing Tensor struct.

---

## License

MIT -- do whatever you want with it.
