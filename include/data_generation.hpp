#pragma once

#include <string>
#include <vector>
#include "point.hpp"

namespace data {

/// Generate `num` uniformly random 2D points in [min, max]^2.
/// @param seed  Fixed seed for reproducibility (default 100).
std::vector<Point> generate_points(int num, float min, float max, int seed = 100);

/// Generate `num` points drawn from an isotropic 2D Gaussian.
/// @param center   Mean of the distribution.
/// @param std_dev  Standard deviation on both axes (must be > 0).
/// @param seed     Fixed seed for reproducibility (default 100).
std::vector<Point> generate_gaussian_points(int num, Point center, float std_dev, int seed = 100);

/// Generate `num` points spread (round-robin) over `clusters` Gaussian blobs
/// whose centers are picked uniformly in [min, max]^2.  Points are clamped to
/// [min, max] so they always stay inside the plot.
std::vector<Point> generate_clustered_points(int num, int clusters, float std_dev,
                                             float min, float max, int seed = 100);

/// Append `count` outliers to `points`.  Each one is placed close to a random
/// corner of [min, max]^2, so it is far from the rest of the data.
void add_outliers(std::vector<Point>& points, int count, float min, float max, int seed = 100);

/// Save points as "x,y" CSV (with a header row).
void save_points_csv(const std::vector<Point>& points, const std::string& filename);

/// Load points written by save_points_csv.
std::vector<Point> load_points_csv(const std::string& filename);

/// Print every point to stdout.
void print_points(const std::vector<Point>& points);

/// In-place Fisher-Yates shuffle.
void shuffle_points(std::vector<Point>& points);

} // namespace data
