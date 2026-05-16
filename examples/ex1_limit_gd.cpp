#include "data_generation.hpp"
#include "optimization.hpp"
#include "visualization.hpp"

// ---------------------------------------------------------------------------
// Lecture 1 — Gradient Descent via the Limit (Finite-Difference) Form
//
// Shows how the derivative definition ∂L/∂x ≈ [L(x+h) − L(x)] / h can be
// used directly as a gradient estimate, without ever deriving the closed-form
// expression by hand.
// ---------------------------------------------------------------------------

int main() {
    constexpr int   N      = 100;
    constexpr float MIN    = 0.0f;
    constexpr float MAX    = 100.0f;
    constexpr int   EPOCHS = 100;
    constexpr float H      = 0.001f;   // finite-difference step
    constexpr float LR     = 5.0f;    // learning rate

    auto points = data::generate_points(N, MIN, MAX);

    Point p = {1.0f, 1.0f};
    auto losses = optim::gradient_descent_limit(points, p, EPOCHS, H, LR);

    // Append the converged point so plot_scatter can highlight it in red
    points.push_back(p);

    viz::PlotConfig scatter_cfg;
    scatter_cfg.width        = 800;
    scatter_cfg.height       = 800;
    scatter_cfg.val_min      = MIN;
    scatter_cfg.val_max      = MAX;
    scatter_cfg.title        = "GD (Limit Form) - Optimal Point";
    scatter_cfg.mark_last_red = true;
    scatter_cfg.show_legend  = true;
    viz::plot_scatter(points, scatter_cfg);

    viz::PlotConfig loss_cfg;
    loss_cfg.width   = 800;
    loss_cfg.height  = 500;
    loss_cfg.val_min = 0.0f;
    loss_cfg.val_max = static_cast<float>(EPOCHS - 1);
    loss_cfg.title   = "GD (Limit Form) - Loss Curve";
    viz::plot_loss_curve(losses, loss_cfg);

    return 0;
}
