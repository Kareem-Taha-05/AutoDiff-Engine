#include "data_generation.hpp"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <stdexcept>

namespace data {

std::vector<Point> generate_points(int num, float min, float max, int seed) {
    if (num <= 0)
        throw std::invalid_argument("num must be > 0");
    if (min >= max)
        throw std::invalid_argument("min must be < max");

    std::srand(static_cast<unsigned>(seed));

    std::vector<Point> pts(static_cast<std::size_t>(num));
    for (auto& p : pts) {
        p.x = min + static_cast<float>(std::rand()) / RAND_MAX * (max - min);
        p.y = min + static_cast<float>(std::rand()) / RAND_MAX * (max - min);
    }
    return pts;
}

void print_points(const std::vector<Point>& points) {
    for (std::size_t i = 0; i < points.size(); ++i) {
        std::cout << "point(" << (i + 1) << "): ("
                  << points[i].x << ", " << points[i].y << ")\n";
    }
}

void shuffle_points(std::vector<Point>& points) {
    // Fisher-Yates shuffle — same algorithm as original, kept custom per design goal
    for (int i = static_cast<int>(points.size()) - 1; i > 0; --i) {
        int j = std::rand() % (i + 1);
        std::swap(points[static_cast<std::size_t>(i)],
                  points[static_cast<std::size_t>(j)]);
    }
}

} // namespace data
