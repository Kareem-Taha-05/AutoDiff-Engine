#include "data_generation.hpp"

#include <iostream>
#include <fstream>
#include <random>
#include <cstdint>
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

// ---------------------------------------------------------------------------
// Dataset generators
//
// These use their own std::mt19937 instead of the global std::rand state, so
// generating a dataset never disturbs the shuffle sequence of an optimizer,
// and std::normal_distribution gives us real Gaussians.
// ---------------------------------------------------------------------------

std::vector<Point> generate_gaussian_points(int num, Point center, float std_dev, int seed) {
    if (num <= 0)
        throw std::invalid_argument("num must be > 0");
    if (std_dev <= 0.0f)
        throw std::invalid_argument("std_dev must be > 0");

    std::mt19937 gen(static_cast<std::uint32_t>(seed));
    std::normal_distribution<float> noise(0.0f, std_dev);

    std::vector<Point> pts(static_cast<std::size_t>(num));
    for (auto& p : pts) {
        p.x = center.x + noise(gen);
        p.y = center.y + noise(gen);
    }
    return pts;
}

std::vector<Point> generate_clustered_points(int num, int clusters, float std_dev,
                                             float min, float max, int seed)
{
    if (num <= 0)
        throw std::invalid_argument("num must be > 0");
    if (clusters <= 0)
        throw std::invalid_argument("clusters must be > 0");
    if (std_dev <= 0.0f)
        throw std::invalid_argument("std_dev must be > 0");
    if (min >= max)
        throw std::invalid_argument("min must be < max");

    std::mt19937 gen(static_cast<std::uint32_t>(seed));
    std::uniform_real_distribution<float> center_dist(min, max);
    std::normal_distribution<float> noise(0.0f, std_dev);

    // Pick the center of every cluster first
    std::vector<Point> centers(static_cast<std::size_t>(clusters));
    for (auto& c : centers) {
        c.x = center_dist(gen);
        c.y = center_dist(gen);
    }

    std::vector<Point> pts(static_cast<std::size_t>(num));
    for (std::size_t i = 0; i < pts.size(); ++i) {
        const Point& c = centers[i % centers.size()];   // round-robin -> equal-sized clusters
        pts[i].x = std::clamp(c.x + noise(gen), min, max);
        pts[i].y = std::clamp(c.y + noise(gen), min, max);
    }
    return pts;
}

void add_outliers(std::vector<Point>& points, int count, float min, float max, int seed) {
    if (count < 0)
        throw std::invalid_argument("count must be >= 0");
    if (min >= max)
        throw std::invalid_argument("min must be < max");

    // seed + 1 so the outliers don't replay the numbers the dataset used
    std::mt19937 gen(static_cast<std::uint32_t>(seed) + 1u);
    std::uniform_int_distribution<int> coin(0, 1);
    std::uniform_real_distribution<float> jitter(0.0f, 0.05f * (max - min));

    for (int i = 0; i < count; ++i) {
        Point p;
        p.x = coin(gen) ? (max - jitter(gen)) : (min + jitter(gen));
        p.y = coin(gen) ? (max - jitter(gen)) : (min + jitter(gen));
        points.push_back(p);
    }
}

void save_points_csv(const std::vector<Point>& points, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("could not open " + filename);

    file << "x,y\n";
    for (const auto& p : points)
        file << p.x << "," << p.y << "\n";
}

std::vector<Point> load_points_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("could not open " + filename);

    std::vector<Point> pts;
    std::string line;
    std::getline(file, line);                       // skip the header

    while (std::getline(file, line)) {
        const std::size_t comma = line.find(',');
        if (comma == std::string::npos) continue;   // ignore blank / malformed lines

        Point p;
        p.x = std::stof(line.substr(0, comma));
        p.y = std::stof(line.substr(comma + 1));
        pts.push_back(p);
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
