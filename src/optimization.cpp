#include "optimization.hpp"
#include "data_generation.hpp"

#include <cmath>
#include <algorithm>
#include <ctime>
#include <stdexcept>

namespace optim {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr float kDistEps = 1e-8f;  // guard against division by zero

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

} // namespace optim
