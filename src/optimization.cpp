#include "optimization.hpp"
#include "data_generation.hpp"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

namespace optim {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr float kDistEps = 1e-8f;  // guard against division by zero

// Lecture 5 hyper-parameters (the usual defaults from the papers)
static constexpr float kBeta        = 0.9f;     // momentum / RMSprop / Adam first-moment decay
static constexpr float kBeta2       = 0.999f;   // Adam second-moment decay
static constexpr float kAdaptiveEps = 1e-8f;    // keeps sqrt(v) from dividing by ~0
static constexpr float kPi          = 3.14159265358979f;

// ---------------------------------------------------------------------------
// Loss
// ---------------------------------------------------------------------------

float euclidean_loss(const std::vector<Point>& pts, const Point& p) {
    float loss = 0.0f;
    for (const auto& pt : pts) {
        loss += std::sqrt(
            std::pow(pt.x - p.x, 2.0f) +
            std::pow(pt.y - p.y, 2.0f));
    }
    return loss / static_cast<float>(pts.size());
}

// ---------------------------------------------------------------------------
// Full-batch GD — numerical gradient (limit / finite-difference form)
//
// The derivative is approximated as:
//   ∂L/∂x ≈ [ L(p + h·eₓ) − L(p) ] / h
// This is the definition of the derivative taken to its limit — hence the
// name "limit form".  It is O(3·N) loss evaluations per epoch.
// ---------------------------------------------------------------------------
std::vector<Point> gradient_descent_limit(
    const std::vector<Point>& pts,
    Point& p,
    int epochs,
    float h,
    float lr)
{
    if (h <= 0.0f)
        throw std::invalid_argument("Finite-difference step h must be positive");
    if (epochs <= 0)
        throw std::invalid_argument("epochs must be > 0");

    std::vector<Point> losses(static_cast<std::size_t>(epochs));

    for (int i = 0; i < epochs; ++i) {
        float base_loss = euclidean_loss(pts, p);
        losses[static_cast<std::size_t>(i)] = {static_cast<float>(i), base_loss};

        // Perturbed points along each axis
        Point px = {p.x + h, p.y};
        Point py = {p.x,     p.y + h};

        float dloss_dx = (euclidean_loss(pts, px) - base_loss) / h;
        float dloss_dy = (euclidean_loss(pts, py) - base_loss) / h;

        p.x -= lr * dloss_dx;
        p.y -= lr * dloss_dy;
    }

    return losses;
}

// ---------------------------------------------------------------------------
// Full-batch GD — analytical gradient (closed form)
//
// Exact partial derivative of the mean Euclidean distance loss:
//   ∂L/∂x = −(1/N) · Σ (xᵢ − x) / ||pᵢ − p||₂
//   ∂L/∂y = −(1/N) · Σ (yᵢ − y) / ||pᵢ − p||₂
//
// O(N) per epoch — no extra loss evaluations needed.
// ---------------------------------------------------------------------------
std::vector<Point> gradient_descent_closed(
    const std::vector<Point>& pts,
    Point& p,
    int epochs,
    float lr)
{
    if (epochs <= 0)
        throw std::invalid_argument("epochs must be > 0");

    const float n = static_cast<float>(pts.size());
    std::vector<Point> losses(static_cast<std::size_t>(epochs));

    for (int i = 0; i < epochs; ++i) {
        float dloss_dx = 0.0f;
        float dloss_dy = 0.0f;

        losses[static_cast<std::size_t>(i)] = {static_cast<float>(i), euclidean_loss(pts, p)};

        for (const auto& pt : pts) {
            float dist = std::sqrt(
                std::pow(pt.x - p.x, 2.0f) +
                std::pow(pt.y - p.y, 2.0f));

            if (dist < kDistEps) continue;  // skip coincident point — gradient undefined

            float inv_dist = 1.0f / dist;
            dloss_dx += inv_dist * (pt.x - p.x);
            dloss_dy += inv_dist * (pt.y - p.y);
        }

        // Negate: the closed-form sum above equals −∂L/∂x before the divide
        dloss_dx = -(dloss_dx / n);
        dloss_dy = -(dloss_dy / n);

        p.x -= lr * dloss_dx;
        p.y -= lr * dloss_dy;
    }

    return losses;
}

// ---------------------------------------------------------------------------
// Mini-batch SGD
//
// Same closed-form gradient as above, but evaluated on a randomly shuffled
// mini-batch of size `batch_size` at every inner step.
// The epoch loss is the mean of all batch losses in that epoch.
// ---------------------------------------------------------------------------
std::vector<Point> stochastic_gradient_descent(
    std::vector<Point> pts,   // local copy — we shuffle in-place
    Point& p,
    int epochs,
    int batch_size,
    float lr)
{
    if (epochs <= 0)    throw std::invalid_argument("epochs must be > 0");
    if (batch_size <= 0) throw std::invalid_argument("batch_size must be > 0");

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const int n_pts     = static_cast<int>(pts.size());
    const int n_batches = static_cast<int>(
        std::ceil(static_cast<float>(n_pts) / batch_size));

    std::vector<Point> losses(static_cast<std::size_t>(epochs));

    for (int i = 0; i < epochs; ++i) {
        data::shuffle_points(pts);
        float epoch_loss_sum = 0.0f;

        for (int j = 0; j < n_batches; ++j) {
            int start             = j * batch_size;
            int end               = std::min(start + batch_size, n_pts);
            int current_batch_sz  = end - start;

            // Record this batch's loss contribution
            std::vector<Point> batch(
                pts.begin() + start, pts.begin() + end);
            epoch_loss_sum += euclidean_loss(batch, p);

            // Compute gradient on the batch
            float dloss_dx = 0.0f;
            float dloss_dy = 0.0f;

            for (int z = start; z < end; ++z) {
                float dist = std::sqrt(
                    std::pow(pts[static_cast<std::size_t>(z)].x - p.x, 2.0f) +
                    std::pow(pts[static_cast<std::size_t>(z)].y - p.y, 2.0f));

                if (dist < kDistEps) continue;

                float inv_dist = 1.0f / dist;
                dloss_dx += inv_dist * (pts[static_cast<std::size_t>(z)].x - p.x);
                dloss_dy += inv_dist * (pts[static_cast<std::size_t>(z)].y - p.y);
            }

            dloss_dx = -(dloss_dx / static_cast<float>(current_batch_sz));
            dloss_dy = -(dloss_dy / static_cast<float>(current_batch_sz));

            p.x -= lr * dloss_dx;
            p.y -= lr * dloss_dy;
        }

        losses[static_cast<std::size_t>(i)] = {
            static_cast<float>(i),
            epoch_loss_sum / static_cast<float>(n_batches)
        };
    }

    return losses;
}

// ===========================================================================
// Lecture 5 — optimizer comparison
// ===========================================================================

// ---------------------------------------------------------------------------
// Update rules
//
// One function per parameter keeps every rule to a few lines, and lets the
// dataset optimizer and the analytic-function optimizer share the same code.
// ---------------------------------------------------------------------------

std::string optimizer_name(Optimizer type) {
    switch (type) {
        case Optimizer::Momentum: return "momentum";
        case Optimizer::AdaGrad:  return "adagrad";
        case Optimizer::RMSprop:  return "rmsprop";
        case Optimizer::Adam:     return "adam";
        case Optimizer::GD:       return "gd";
    }
    return "unknown";
}

float step_1d(float grad, float& m, float& v, int t, Optimizer type, float lr) {
    switch (type) {

        case Optimizer::Momentum:
            // Gradient history: old gradients fade out by kBeta every update
            m = kBeta * m + grad;
            return lr * m;

        case Optimizer::AdaGrad:
            // v only ever grows, so the effective step size only ever shrinks
            v += grad * grad;
            return lr * grad / (std::sqrt(v) + kAdaptiveEps);

        case Optimizer::RMSprop:
            // AdaGrad, but old squared gradients fade out -> step size can recover
            v = kBeta * v + (1.0f - kBeta) * grad * grad;
            return lr * grad / (std::sqrt(v) + kAdaptiveEps);

        case Optimizer::Adam: {
            m = kBeta  * m + (1.0f - kBeta)  * grad;          // momentum part
            v = kBeta2 * v + (1.0f - kBeta2) * grad * grad;   // RMSprop part

            // m and v start at 0, so they are biased towards 0 in the first
            // updates; dividing by (1 - beta^t) undoes that.
            const float tf    = static_cast<float>(t);
            const float m_hat = m / (1.0f - std::pow(kBeta,  tf));
            const float v_hat = v / (1.0f - std::pow(kBeta2, tf));
            return lr * m_hat / (std::sqrt(v_hat) + kAdaptiveEps);
        }

        case Optimizer::GD:
            break;
    }
    return lr * grad;   // plain gradient descent
}

void apply_step(Point& p, const Point& grad, OptimizerState& state, Optimizer type, float lr) {
    ++state.t;
    p.x -= step_1d(grad.x, state.m.x, state.v.x, state.t, type, lr);
    p.y -= step_1d(grad.y, state.m.y, state.v.y, state.t, type, lr);
}

// ---------------------------------------------------------------------------
// Gradient of the mean Euclidean distance over pts[first, last)
// ---------------------------------------------------------------------------
Point euclidean_gradient(const std::vector<Point>& pts,
                         std::size_t first, std::size_t last, const Point& p)
{
    if (first >= last || last > pts.size())
        throw std::invalid_argument("euclidean_gradient: invalid range");

    Point grad;

    for (std::size_t z = first; z < last; ++z) {
        const float dx = pts[z].x - p.x;
        const float dy = pts[z].y - p.y;
        const float dist = std::sqrt(dx * dx + dy * dy);

        if (dist < kDistEps) continue;  // coincident point — gradient undefined

        const float inv_dist = 1.0f / dist;
        grad.x += inv_dist * dx;
        grad.y += inv_dist * dy;
    }

    const float count = static_cast<float>(last - first);
    grad.x = -(grad.x / count);
    grad.y = -(grad.y / count);
    return grad;
}

// ---------------------------------------------------------------------------
// Mini-batch training with a selectable optimizer
// ---------------------------------------------------------------------------
OptimizerResult run_optimizer(
    std::vector<Point> pts,     // by value — shuffled in place
    Point start,                // by value — every optimizer starts from the same place
    Optimizer type,
    int epochs,
    int batch_size,
    float lr,
    int seed,
    float tolerance)
{
    if (pts.empty())     throw std::invalid_argument("pts must not be empty");
    if (epochs <= 0)     throw std::invalid_argument("epochs must be > 0");
    if (batch_size <= 0) throw std::invalid_argument("batch_size must be > 0");
    if (lr <= 0.0f)      throw std::invalid_argument("lr must be > 0");

    OptimizerResult result;
    result.name = optimizer_name(type);

    Point p = start;
    OptimizerState state;

    const std::size_t n_pts = pts.size();
    const std::size_t bs    = static_cast<std::size_t>(batch_size);
    const std::size_t n_batches = (n_pts + bs - 1) / bs;

    result.initial_loss = euclidean_loss(pts, p);
    float previous_loss = result.initial_loss;

    // Same seed => data::shuffle_points produces the same batches for every optimizer
    std::srand(static_cast<unsigned>(seed));

    const auto begin = std::chrono::steady_clock::now();

    for (int i = 0; i < epochs; ++i) {
        data::shuffle_points(pts);

        for (std::size_t j = 0; j < n_batches; ++j) {
            const std::size_t first = j * bs;
            const std::size_t last  = std::min(first + bs, n_pts);

            const Point grad = euclidean_gradient(pts, first, last, p);
            apply_step(p, grad, state, type, lr);

            result.trajectory.push_back(p);
            ++result.updates;
        }

        // Loss of the WHOLE dataset at the end of the epoch.  The mean of the
        // batch losses mixes different samples and different positions of p,
        // so it would not be comparable between optimizers.
        const float epoch_loss = euclidean_loss(pts, p);
        result.losses.push_back({static_cast<float>(i), epoch_loss});

        if (std::fabs(previous_loss - epoch_loss) < tolerance) {
            result.epochs_to_converge = i + 1;
            break;
        }
        previous_loss = epoch_loss;
    }

    const auto end = std::chrono::steady_clock::now();
    result.time_ms    = std::chrono::duration<double, std::milli>(end - begin).count();
    result.final_loss = result.losses.back().y;

    return result;
}

// ---------------------------------------------------------------------------
// Non-convex objectives
// ---------------------------------------------------------------------------

float objective_loss(Objective objective, const Point& p) {
    if (objective == Objective::Rosenbrock) {
        const float a = 1.0f - p.x;
        const float b = p.y - p.x * p.x;
        return a * a + 100.0f * b * b;
    }
    // Rastrigin
    return 20.0f
         + (p.x * p.x - 10.0f * std::cos(2.0f * kPi * p.x))
         + (p.y * p.y - 10.0f * std::cos(2.0f * kPi * p.y));
}

Point objective_gradient(Objective objective, const Point& p) {
    Point grad;
    if (objective == Objective::Rosenbrock) {
        const float b = p.y - p.x * p.x;
        grad.x = -2.0f * (1.0f - p.x) - 400.0f * p.x * b;
        grad.y = 200.0f * b;
        return grad;
    }
    // Rastrigin
    grad.x = 2.0f * p.x + 20.0f * kPi * std::sin(2.0f * kPi * p.x);
    grad.y = 2.0f * p.y + 20.0f * kPi * std::sin(2.0f * kPi * p.y);
    return grad;
}

OptimizerResult run_analytic_optimizer(
    Objective objective,
    Point start,
    Optimizer type,
    int epochs,
    float lr,
    float tolerance)
{
    if (epochs <= 0) throw std::invalid_argument("epochs must be > 0");
    if (lr <= 0.0f)  throw std::invalid_argument("lr must be > 0");

    OptimizerResult result;
    result.name = optimizer_name(type);

    Point p = start;
    OptimizerState state;

    result.initial_loss = objective_loss(objective, p);
    float previous_loss = result.initial_loss;

    const auto begin = std::chrono::steady_clock::now();

    for (int i = 0; i < epochs; ++i) {
        const Point grad = objective_gradient(objective, p);
        apply_step(p, grad, state, type, lr);

        result.trajectory.push_back(p);
        ++result.updates;

        const float loss = objective_loss(objective, p);
        result.losses.push_back({static_cast<float>(i), loss});

        // Learning rate too large for this landscape: stop instead of plotting garbage
        if (!std::isfinite(loss)) break;

        if (std::fabs(previous_loss - loss) < tolerance) {
            result.epochs_to_converge = i + 1;
            break;
        }
        previous_loss = loss;
    }

    const auto end = std::chrono::steady_clock::now();
    result.time_ms    = std::chrono::duration<double, std::milli>(end - begin).count();
    result.final_loss = result.losses.back().y;

    return result;
}

} // namespace optim
