#pragma once

#include <vector>
#include <stdexcept>
#include "point.hpp"

// ---------------------------------------------------------------------------
// optim — gradient descent variants over the mean Euclidean distance loss.
//
//   Loss(p) = (1/N) * Σ ||p_i − p||₂
//
// All three variants minimise the same scalar objective; they differ in how
// the gradient is estimated and how many samples are used per update step.
// ---------------------------------------------------------------------------
namespace optim {

// ---- loss ------------------------------------------------------------------

/// Mean Euclidean distance from every point in `pts` to `p`.
float euclidean_loss(const std::vector<Point>& pts, const Point& p);

// ---- full-batch GD — numerical gradient (limit form) ----------------------

/// Estimate ∂L/∂x and ∂L/∂y via the finite-difference definition of the
/// derivative: f'(x) ≈ [f(x+h) − f(x)] / h.
///
/// @param h   Step size for the finite-difference approximation (e.g. 0.001).
/// @param lr  Learning rate (called delta in the lecture notes).
/// @return    Vector of (epoch, loss) pairs for plotting.
std::vector<Point> gradient_descent_limit(
    const std::vector<Point>& pts,
    Point& p,
    int epochs,
    float h,
    float lr);

// ---- full-batch GD — analytical gradient (closed form) --------------------

/// Use the exact partial derivatives of the mean Euclidean distance:
///   ∂L/∂x = −(1/N) * Σ (x_i − x) / ||p_i − p||₂
///
/// Skips points where the distance underflows to avoid NaN.
///
/// @return  Vector of (epoch, loss) pairs for plotting.
std::vector<Point> gradient_descent_closed(
    const std::vector<Point>& pts,
    Point& p,
    int epochs,
    float lr);

// ---- mini-batch SGD --------------------------------------------------------

/// Same closed-form gradient as above, but computed on a randomly shuffled
/// mini-batch of size `batch_size` at every update step.
///
/// @return  Vector of (epoch, mean-batch-loss) pairs for plotting.
std::vector<Point> stochastic_gradient_descent(
    std::vector<Point> pts,     // intentional copy — will be shuffled
    Point& p,
    int epochs,
    int batch_size,
    float lr);

} // namespace optim
