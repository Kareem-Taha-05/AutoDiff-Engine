#pragma once

/// A simple 2D point. Used as both a data sample and a (epoch, loss) pair
/// when tracking optimization history.
struct Point {
    float x = 0.0f;
    float y = 0.0f;
};
