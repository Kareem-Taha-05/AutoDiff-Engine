#pragma once

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
    bool  show_grad_vec  = false;  ///< TODO: overlay gradient direction arrow
};

/// Scatter plot.  If `mark_last_red` is true the last point is drawn in red
/// (convention: append the optimal point to the vector before calling).
void plot_scatter(const std::vector<Point>& points, const PlotConfig& cfg = {});

/// Line plot of a loss curve — connects (epoch, loss) pairs with lines and
/// also draws individual sample dots.
void plot_loss_curve(const std::vector<Point>& losses, const PlotConfig& cfg = {});

} // namespace viz
