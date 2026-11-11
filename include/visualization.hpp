#pragma once

#include <functional>
#include <vector>
#include <string>
#include "point.hpp"

// ---------------------------------------------------------------------------
// viz — Raylib-based plotting utilities.
//
// Each function opens its own window, runs the render loop until the user
// closes it, then cleans up.  Call them sequentially — Raylib's context is
// not reentrant.
// ---------------------------------------------------------------------------
namespace viz {

/// Configuration for a single plot window.
struct PlotConfig {
    int   width          = 800;
    int   height         = 800;
    float val_min        = 0.0f;
    float val_max        = 100.0f;
    std::string title    = "Plot";
    bool  mark_last_red  = true;   ///< highlight the last point (optimal p) in red
    bool  show_legend    = true;
    bool  show_grad_vec  = false;  ///< contour plot: start with the gradient field visible (hold G to show it either way)
};

/// Loss and gradient as functions of a single point, so the contour plot
/// doesn't need to know where the loss comes from (dataset or analytic function).
using LossFn = std::function<float(const Point&)>;
using GradFn = std::function<Point(const Point&)>;

/// One optimizer path to draw on top of the contour map.
struct Trajectory {
    std::string        name;
    std::vector<Point> path;   ///< position after every update
};

/// Scatter plot.  If `mark_last_red` is true the last point is drawn in red
/// (convention: append the optimal point to the vector before calling).
void plot_scatter(const std::vector<Point>& points, const PlotConfig& cfg = {});

/// Line plot of a loss curve — connects (epoch, loss) pairs with lines and
/// also draws individual sample dots.
void plot_loss_curve(const std::vector<Point>& losses, const PlotConfig& cfg = {});

/// Loss landscape as a banded heatmap (blue = low, red = high; every band is
/// one contour level) with every trajectory drawn on top.  Hold G to overlay the
/// gradient field: one arrow per grid spot pointing along -grad(L), i.e. where
/// gradient descent would step next (direction only, not magnitude).
///
/// val_min / val_max in `cfg` give the range of BOTH axes.
void plot_loss_contour(const LossFn& loss,
                       const GradFn& grad,
                       const std::vector<Trajectory>& trajectories,
                       const PlotConfig& cfg = {});

} // namespace viz
