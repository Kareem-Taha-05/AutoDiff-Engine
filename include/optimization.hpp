#pragma once

#include <cstddef>
#include <string>
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
//
// Lecture 5 adds an optimizer comparison layer on top: Momentum, AdaGrad,
// RMSprop and Adam share one update interface, and the same optimizers can
// run on non-convex 2D test functions (Rosenbrock, Rastrigin).
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

// ===========================================================================
// Lecture 5 — optimizer comparison
// ===========================================================================

/// The update rules that can be compared head to head.
/// TODO: Nesterov momentum (needs the gradient at a lookahead position, so it
///       can't share the single-step interface below).
enum class Optimizer { GD, Momentum, AdaGrad, RMSprop, Adam };

/// 2D objectives that don't depend on a dataset.
enum class Objective {
    Rosenbrock,   ///< (1-x)^2 + 100(y-x^2)^2   — minimum 0 at (1, 1), curved valley
    Rastrigin     ///< 20 + sum(x^2 - 10cos(2πx)) — minimum 0 at (0, 0), many local minima
};

/// The memory an optimizer carries between two updates.
struct OptimizerState {
    Point m;      ///< running average of the gradient (Momentum, Adam)
    Point v;      ///< running sum / average of the squared gradient (AdaGrad, RMSprop, Adam)
    int   t = 0;  ///< updates done so far (Adam's bias correction needs it)
};

/// Everything we want to know about one run, so runs can be compared.
struct OptimizerResult {
    std::string        name;                  ///< optimizer name, e.g. "adam"
    std::vector<Point> losses;                ///< (epoch, loss) pairs for plotting
    std::vector<Point> trajectory;            ///< position of p after every update
    float  initial_loss       = 0.0f;
    float  final_loss         = 0.0f;
    int    epochs_to_converge = -1;           ///< -1 if the tolerance was never reached
    int    updates            = 0;            ///< number of parameter updates
    double time_ms            = 0.0;          ///< wall-clock training time
};

/// Lower-case display name of an optimizer ("gd", "momentum", ...).
std::string optimizer_name(Optimizer type);

/// Update rule of one optimizer for ONE parameter.
/// @param grad  Gradient of that parameter.
/// @param m,v   The parameter's slots in OptimizerState (modified in place).
/// @param t     1-based update counter (Adam's bias correction).
/// @return      The amount to SUBTRACT from the parameter.
float step_1d(float grad, float& m, float& v, int t, Optimizer type, float lr);

/// Apply one update to both coordinates of `p`.
void apply_step(Point& p, const Point& grad, OptimizerState& state, Optimizer type, float lr);

/// Gradient of the mean Euclidean distance over pts[first, last).
/// Same math as gradient_descent_closed, exposed as a reusable building block.
/// Points coinciding with `p` are skipped (gradient undefined there).
Point euclidean_gradient(const std::vector<Point>& pts,
                         std::size_t first, std::size_t last, const Point& p);

/// Mini-batch training of `p` with the chosen optimizer on the mean Euclidean
/// distance loss.
///
/// `pts` and `start` are taken BY VALUE on purpose: pts is shuffled, and every
/// optimizer must begin from the same data and the same start point.  The
/// shuffle is seeded with `seed`, so every optimizer sees identical batches.
/// The per-epoch loss is measured on the WHOLE dataset, so curves of different
/// optimizers are comparable.
///
/// @param tolerance  Converged when |loss(epoch) - loss(epoch-1)| < tolerance
///                   (the run stops early at that point).
OptimizerResult run_optimizer(
    std::vector<Point> pts,
    Point start,
    Optimizer type,
    int epochs,
    int batch_size,
    float lr,
    int seed,
    float tolerance);

/// Value of a dataset-free 2D objective at `p`.
float objective_loss(Objective objective, const Point& p);

/// Exact gradient of a dataset-free 2D objective at `p`.
Point objective_gradient(Objective objective, const Point& p);

/// Full-batch optimization of a 2D objective (one epoch = one update).
/// Stops early if the loss stops being finite (learning rate too large).
OptimizerResult run_analytic_optimizer(
    Objective objective,
    Point start,
    Optimizer type,
    int epochs,
    float lr,
    float tolerance);

} // namespace optim
