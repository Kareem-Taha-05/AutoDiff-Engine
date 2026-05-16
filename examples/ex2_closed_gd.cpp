#include "data_generation.hpp"
#include "optimization.hpp"
#include "visualization.hpp"

// ---------------------------------------------------------------------------
// Lecture 2 — Gradient Descent via the Closed-Form (Analytical) Gradient
//
// Derives the partial derivatives of the mean Euclidean distance by hand and
// uses them directly — no finite-difference approximation needed.
// Compare convergence speed with ex1_limit_gd.
// ---------------------------------------------------------------------------

int main() {
    constexpr int   N      = 100;
    constexpr float MIN    = 0.0f;
    constexpr float MAX    = 100.0f;
    constexpr int   EPOCHS = 100;
    constexpr float LR     = 5.0f;

    auto points = data::generate_points(N, MIN, MAX);

    Point p = {1.0f, 1.0f};
    auto losses = optim::gradient_descent_closed(points, p, EPOCHS, LR);

    points.push_back(p);

    viz::PlotConfig scatter_cfg;
    scatter_cfg.width        = 800;
    scatter_cfg.height       = 800;
    scatter_cfg.val_min      = MIN;
    scatter_cfg.val_max      = MAX;
    scatter_cfg.title        = "GD (Closed Form) - Optimal Point";
    scatter_cfg.mark_last_red = true;
    scatter_cfg.show_legend  = true;
    viz::plot_scatter(points, scatter_cfg);

    viz::PlotConfig loss_cfg;
    loss_cfg.width   = 800;
    loss_cfg.height  = 500;
    loss_cfg.val_min = 0.0f;
    loss_cfg.val_max = static_cast<float>(EPOCHS - 1);
    loss_cfg.title   = "GD (Closed Form) - Loss Curve";
    viz::plot_loss_curve(losses, loss_cfg);

    return 0;
}
