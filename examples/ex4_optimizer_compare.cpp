#include "data_generation.hpp"
#include "optimization.hpp"
#include "report.hpp"
#include "visualization.hpp"

#include <vector>

// ---------------------------------------------------------------------------
// Lecture 5 — Optimizer Comparison
//
// GD, Momentum, AdaGrad, RMSprop and Adam run on exactly the same dataset,
// start point and mini-batch sequence, so the only thing that changes is the
// update rule.  The dataset has 3 clusters plus a few far-away outliers, which
// is what makes the geometric median interesting.
//
// Output: a console report, experiment_report.csv, loss_history.csv,
// dataset.csv, and three Raylib windows (scatter, contour map, loss curve).
// ---------------------------------------------------------------------------

int main() {
    constexpr int   N           = 100;
    constexpr float MIN         = 0.0f;
    constexpr float MAX         = 100.0f;
    constexpr int   SEED        = 100;
    constexpr int   CLUSTERS    = 3;
    constexpr float CLUSTER_STD = 8.0f;
    constexpr int   OUTLIERS    = 5;

    constexpr int   EPOCHS      = 100;
    constexpr int   BATCH_SIZE  = 10;
    constexpr float LR          = 1.0f;
    constexpr float TOLERANCE   = 0.001f;   // converged when the epoch loss changes by less than this

    auto points = data::generate_clustered_points(N, CLUSTERS, CLUSTER_STD, MIN, MAX, SEED);
    data::add_outliers(points, OUTLIERS, MIN, MAX, SEED);
    data::save_points_csv(points, "dataset.csv");

    const Point start = {1.0f, 1.0f};

    const std::vector<optim::Optimizer> optimizers = {
        optim::Optimizer::GD,
        optim::Optimizer::Momentum,
        optim::Optimizer::AdaGrad,
        optim::Optimizer::RMSprop,
        optim::Optimizer::Adam
    };

    std::vector<optim::OptimizerResult> results;
    for (auto type : optimizers) {
        results.push_back(optim::run_optimizer(points, start, type,
                                               EPOCHS, BATCH_SIZE, LR, SEED, TOLERANCE));
    }

    report::print(results);
    report::save_summary_csv(results, "experiment_report.csv");
    report::save_loss_history_csv(results, "loss_history.csv");

    // ---- scatter: dataset + where Adam ended up ----------------------------
    auto scatter_points = points;
    scatter_points.push_back(results.back().trajectory.back());

    viz::PlotConfig scatter_cfg;
    scatter_cfg.width         = 800;
    scatter_cfg.height        = 800;
    scatter_cfg.val_min       = MIN;
    scatter_cfg.val_max       = MAX;
    scatter_cfg.title         = "Clusters + Outliers - Optimal Point (Adam)";
    scatter_cfg.mark_last_red = true;
    scatter_cfg.show_legend   = true;
    viz::plot_scatter(scatter_points, scatter_cfg);

    // ---- contour map: loss landscape + every trajectory --------------------
    const viz::LossFn loss_fn = [&points](const Point& q) {
        return optim::euclidean_loss(points, q);
    };
    const viz::GradFn grad_fn = [&points](const Point& q) {
        return optim::euclidean_gradient(points, 0, points.size(), q);
    };

    std::vector<viz::Trajectory> trajectories;
    for (const auto& r : results)
        trajectories.push_back({r.name, r.trajectory});

    viz::PlotConfig contour_cfg;
    contour_cfg.width    = 800;
    contour_cfg.height   = 800;
    contour_cfg.val_min  = MIN;
    contour_cfg.val_max  = MAX;
    contour_cfg.title    = "Loss Contour - Optimizer Trajectories";
    viz::plot_loss_contour(loss_fn, grad_fn, trajectories, contour_cfg);

    // ---- loss curve of the last optimizer (Adam) ---------------------------
    viz::PlotConfig loss_cfg;
    loss_cfg.width   = 800;
    loss_cfg.height  = 500;
    loss_cfg.val_min = 0.0f;
    loss_cfg.val_max = 100.0f;      // loss and epoch share the axis range, both stay below 100 here
    loss_cfg.title   = "Adam - Loss Curve (full-dataset loss per epoch)";
    viz::plot_loss_curve(results.back().losses, loss_cfg);

    return 0;
}
