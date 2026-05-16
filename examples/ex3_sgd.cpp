#include "data_generation.hpp"
#include "optimization.hpp"
#include "visualization.hpp"

// ---------------------------------------------------------------------------
// Lecture 4 — Mini-Batch Stochastic Gradient Descent (SGD)
//
// Instead of computing the full-dataset gradient at every step, SGD estimates
// it from a randomly sampled mini-batch.  Notice the noisy loss curve vs. the
// smooth curves from full-batch GD — that noise is the price of speed.
// ---------------------------------------------------------------------------

int main() {
    constexpr int   N          = 100;
    constexpr float MIN        = 0.0f;
    constexpr float MAX        = 100.0f;
    constexpr int   EPOCHS     = 100;
    constexpr int   BATCH_SIZE = 10;
    constexpr float LR         = 5.0f;

    auto points = data::generate_points(N, MIN, MAX);

    Point p = {1.0f, 1.0f};
    auto losses = optim::stochastic_gradient_descent(points, p, EPOCHS, BATCH_SIZE, LR);

    points.push_back(p);

    viz::PlotConfig scatter_cfg;
    scatter_cfg.width        = 800;
    scatter_cfg.height       = 800;
    scatter_cfg.val_min      = MIN;
    scatter_cfg.val_max      = MAX;
    scatter_cfg.title        = "SGD - Optimal Point";
    scatter_cfg.mark_last_red = true;
    scatter_cfg.show_legend  = true;
    viz::plot_scatter(points, scatter_cfg);

    viz::PlotConfig loss_cfg;
    loss_cfg.width   = 800;
    loss_cfg.height  = 500;
    loss_cfg.val_min = 0.0f;
    loss_cfg.val_max = static_cast<float>(EPOCHS - 1);
    loss_cfg.title   = "SGD - Loss Curve (noisy by design)";
    viz::plot_loss_curve(losses, loss_cfg);

    return 0;
}
