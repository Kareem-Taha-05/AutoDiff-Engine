# AutoDiff Engine

> Gradient Descent and SGD implemented in pure C++17, no ML libraries, no abstraction layers, full math exposed.

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![Build](https://img.shields.io/badge/build-CMake-green)
![Visualization](https://img.shields.io/badge/visualization-Raylib-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What is this?

A personal engineering project where I implement the foundational pieces of ML optimization **completely from scratch in C++**: a strided tensor data structure, gradient descent in two forms (numerical and analytical), and mini-batch SGD, all visualized live using Raylib.

No PyTorch. No Eigen. No NumPy. Just C++, math, and pixels.

I built this to make sure I *genuinely understand* what happens when you call `optimizer.step()`, every multiply, every divide, every gradient accumulation.

---

## What it demonstrates

| Concept | Implementation |
|---|---|
| Custom strided Tensor | `include/tensor.hpp`: Rule of 5, `std::vector` backing, bounds-checked `operator()` |
| Mean Euclidean Distance Loss | `src/optimization.cpp::euclidean_loss()` |
| GD: Numerical gradient | Finite-difference: `[L(x+h) − L(x)] / h` |
| GD: Analytical gradient | Closed-form partial derivatives, exact |
| Mini-batch SGD | Fisher-Yates shuffle + batched gradient accumulation |
| Live Raylib plots | Scrollable/zoomable scatter plots and loss curves with legends |

---

## What you'll see when you run it

**Scatter plot window**
100 random 2D points in black. After optimization converges, the single red dot marks where point `p` ended up, the geometric median of the cloud.

**Loss curve window**
A falling curve from epoch 0 to 99. Full-batch GD (both forms) gives a smooth descent. SGD gives a noisy, jumpier curve, that's the gradient variance from mini-batches. Both converge. That's the point.

Both windows support **mouse-wheel zoom** and stay open until you close them.

---

## Project structure

```
AutoDiff-Engine/
│
├── include/
│   ├── tensor.hpp          # Strided rank-1/2/3 tensor (custom, no Eigen)
│   ├── point.hpp           # Simple 2D point struct
│   ├── optimization.hpp    # GD / SGD algorithm declarations
│   ├── data_generation.hpp # Point generation and shuffling
│   └── visualization.hpp   # Raylib scatter + loss curve plots
│
├── src/
│   ├── optimization.cpp    # All three optimizers, exact math preserved
│   ├── data_generation.cpp # Random point generation, Fisher-Yates shuffle
│   └── visualization.cpp   # Raylib render loops, axis drawing, legends
│
├── examples/
│   ├── ex1_limit_gd.cpp    # GD with finite-difference (numerical) gradient
│   ├── ex2_closed_gd.cpp   # GD with exact analytical gradient
│   └── ex3_sgd.cpp         # Mini-batch SGD
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
```

Each binary opens two Raylib windows in sequence: the scatter plot first, then the loss curve. Close a window with `Esc` or the title-bar button to proceed.

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

### Update rule (all variants)
```
p ← p − lr · ∇L(p)
```

---

## Roadmap / TODOs

These are the open items I'm actively building toward:

- [ ] **3D loss surface:** render an X×Y mesh of loss values as a Raylib 3D surface using `DrawMesh`, so you can visually see the bowl shape and watch the point descend into it.
- [ ] **Gradient direction arrows:** overlay `DrawLineEx` vectors from `p` in the direction of `−∇L` at each epoch to show *where* the gradient is pointing.
- [ ] **Tensor reshape / transpose:** implement `reshape(nx, ny)` returning a view with new strides but shared data (no copy), and a 2D `transpose()`.
- [ ] **Momentum SGD:** add a velocity buffer and β parameter to compare convergence vs. vanilla SGD.
- [ ] **Lecture 5+ content:** extend to linear regression and a from-scratch single-layer network using the existing Tensor struct.

---

## License

MIT -- do whatever you want with it.
