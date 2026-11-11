#include "optimization.hpp"
#include "report.hpp"
#include "visualization.hpp"

#include <vector>

// ---------------------------------------------------------------------------
// Lecture 5 (second experiment) — Non-convex objectives
//
// No dataset here: the optimizers minimise Rosenbrock (a long curved valley)
// or Rastrigin (a grid of local minima) directly.  Every optimizer gets its
// own learning rate, because a rate that suits Adam makes plain GD explode on
// Rosenbrock.  Flip USE_ROSENBROCK to switch landscapes.
// ---------------------------------------------------------------------------

namespace {

struct Run {
    optim::Optimizer type;
    float            lr;
};

} // namespace

int main() {
    constexpr bool USE_ROSENBROCK = true;

    constexpr int   EPOCHS    = 5000;
    constexpr float TOLERANCE = 1e-6f;

    const optim::Objective objective = USE_ROSENBROCK ? optim::Objective::Rosenbrock
                                                      : optim::Objective::Rastrigin;
    const float lo = USE_ROSENBROCK ? -2.0f : -5.12f;
    const float hi = USE_ROSENBROCK ?  2.0f :  5.12f;

    const Point start = USE_ROSENBROCK ? Point{-1.5f, 1.5f} : Point{3.5f, 3.5f};

    const std::vector<Run> runs = USE_ROSENBROCK
        ? std::vector<Run>{
              {optim::Optimizer::GD,       0.001f},
              {optim::Optimizer::Momentum, 0.0002f},
              {optim::Optimizer::AdaGrad,  0.1f},
              {optim::Optimizer::RMSprop,  0.01f},
              {optim::Optimizer::Adam,     0.05f}}
        : std::vector<Run>{
              {optim::Optimizer::GD,       0.005f},
              {optim::Optimizer::Momentum, 0.001f},
              {optim::Optimizer::AdaGrad,  0.3f},
              {optim::Optimizer::RMSprop,  0.02f},
              {optim::Optimizer::Adam,     0.05f}};

    std::vector<optim::OptimizerResult> results;
    for (const auto& run : runs) {
        results.push_back(optim::run_analytic_optimizer(objective, start, run.type,
                                                        EPOCHS, run.lr, TOLERANCE));
    }

    report::print(results);
    report::save_summary_csv(results, "nonconvex_report.csv");
    report::save_loss_history_csv(results, "nonconvex_loss_history.csv");

    const viz::LossFn loss_fn = [objective](const Point& q) {
        return optim::objective_loss(objective, q);
    };
    const viz::GradFn grad_fn = [objective](const Point& q) {
        return optim::objective_gradient(objective, q);
    };

    std::vector<viz::Trajectory> trajectories;
    for (const auto& r : results)
        trajectories.push_back({r.name, r.trajectory});

    viz::PlotConfig cfg;
    cfg.width   = 800;
    cfg.height  = 800;
    cfg.val_min = lo;
    cfg.val_max = hi;
    cfg.title   = USE_ROSENBROCK ? "Rosenbrock - Optimizer Trajectories"
                                 : "Rastrigin - Optimizer Trajectories";
    viz::plot_loss_contour(loss_fn, grad_fn, trajectories, cfg);

    return 0;
}
