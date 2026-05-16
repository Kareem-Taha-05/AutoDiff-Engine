#pragma once

#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>

// ---------------------------------------------------------------------------
// Tensor<T> — a rank-1/2/3 strided tensor backed by std::vector<T>.
//
// Design notes:
//   • Backing store is std::vector<T> instead of raw new[]/delete[].
//     This gives us exception-safe construction, correct copy/move
//     semantics, and RAII destruction for free, so the compiler-generated
//     Rule-of-Five members are all correct.  We still declare them
//     explicitly (= default) to document the intent.
//   • Stride layout mirrors the original (row-major / C-order):
//       1D  stride_x = 1
//       2D  stride_x = y,  stride_y = 1
//       3D  stride_x = y*z, stride_y = z, stride_z = 1
//   • operator() overloads do optional bounds-checking in debug builds.
// ---------------------------------------------------------------------------
template<typename T>
class Tensor {
public:
    // ---- dimensions & strides (public for easy inspection) ----------------
    int length_x{1}, length_y{1}, length_z{1};
    int stride_x{1}, stride_y{0}, stride_z{0};

    // ---- constructors ------------------------------------------------------
    explicit Tensor(int x)
        : length_x(x), length_y(1), length_z(1)
        , stride_x(1), stride_y(0), stride_z(0)
        , data_(static_cast<std::size_t>(x))
    {}

    Tensor(int x, int y)
        : length_x(x), length_y(y), length_z(1)
        , stride_x(y), stride_y(1), stride_z(0)
        , data_(static_cast<std::size_t>(x * y))
    {}

    Tensor(int x, int y, int z)
        : length_x(x), length_y(y), length_z(z)
        , stride_x(y * z), stride_y(z), stride_z(1)
        , data_(static_cast<std::size_t>(x * y * z))
    {}

    // ---- Rule of Five — all correct thanks to std::vector backing ----------
    Tensor(const Tensor&)            = default;
    Tensor(Tensor&&) noexcept        = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor()                        = default;

    // ---- capacity ----------------------------------------------------------
    [[nodiscard]] int size() const noexcept {
        return length_x * length_y * length_z;
    }

    void fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // ---- raw data access (useful for passing to C APIs) --------------------
    T*       raw()       noexcept { return data_.data(); }
    const T* raw() const noexcept { return data_.data(); }

    // ---- element access — operator() ---------------------------------------
    T& operator()(int i) {
        check_1d(i);
        return data_[static_cast<std::size_t>(i * stride_x)];
    }
    T& operator()(int i, int j) {
        check_2d(i, j);
        return data_[static_cast<std::size_t>(i * stride_x + j * stride_y)];
    }
    T& operator()(int i, int j, int k) {
        check_3d(i, j, k);
        return data_[static_cast<std::size_t>(i * stride_x + j * stride_y + k * stride_z)];
    }

    const T& operator()(int i) const {
        check_1d(i);
        return data_[static_cast<std::size_t>(i * stride_x)];
    }
    const T& operator()(int i, int j) const {
        check_2d(i, j);
        return data_[static_cast<std::size_t>(i * stride_x + j * stride_y)];
    }
    const T& operator()(int i, int j, int k) const {
        check_3d(i, j, k);
        return data_[static_cast<std::size_t>(i * stride_x + j * stride_y + k * stride_z)];
    }

    // ---- TODO: reshape (returns a view / new Tensor with same data) --------
    // Tensor<T> reshape(int nx, int ny) const { ... }
    //
    // ---- TODO: transpose 2D ------------------------------------------------
    // Tensor<T> transpose() const { ... }

private:
    std::vector<T> data_;

    void check_1d(int i) const {
#ifndef NDEBUG
        if (i < 0 || i >= length_x)
            throw std::out_of_range(
                "Tensor 1D index " + std::to_string(i) +
                " out of range [0, " + std::to_string(length_x) + ")");
#else
        (void)i;
#endif
    }
    void check_2d(int i, int j) const {
#ifndef NDEBUG
        if (i < 0 || i >= length_x || j < 0 || j >= length_y)
            throw std::out_of_range(
                "Tensor 2D index (" + std::to_string(i) + ", " + std::to_string(j) +
                ") out of range");
#else
        (void)i; (void)j;
#endif
    }
    void check_3d(int i, int j, int k) const {
#ifndef NDEBUG
        if (i < 0 || i >= length_x || j < 0 || j >= length_y || k < 0 || k >= length_z)
            throw std::out_of_range(
                "Tensor 3D index (" + std::to_string(i) + ", " +
                std::to_string(j) + ", " + std::to_string(k) + ") out of range");
#else
        (void)i; (void)j; (void)k;
#endif
    }
};
