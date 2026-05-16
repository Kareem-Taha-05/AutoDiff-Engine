#pragma once

#include <vector>
#include "point.hpp"

namespace data {

/// Generate `num` uniformly random 2D points in [min, max]^2.
/// @param seed  Fixed seed for reproducibility (default 100).
std::vector<Point> generate_points(int num, float min, float max, int seed = 100);

/// Print every point to stdout.
void print_points(const std::vector<Point>& points);

/// In-place Fisher-Yates shuffle.
void shuffle_points(std::vector<Point>& points);

} // namespace data
