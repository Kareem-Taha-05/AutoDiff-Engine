#include "visualization.hpp"
#include "raylib.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <string>

namespace viz {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Map a data-space value to a screen pixel along one axis.
/// offset: the pixel margin on the left/bottom where the axis starts.
static float to_screen(float val, float val_min, float val_max,
                        int pixel_range, int offset, float scale)
{
    return ((val - val_min) / (val_max - val_min)
            * static_cast<float>(pixel_range - offset)) * scale
           + static_cast<float>(offset);
}

/// Draw labelled tick marks on both axes.
static void draw_axes(const PlotConfig& cfg, float scale) {
    const int offset    = 55;
    const int step_px   = static_cast<int>(50.0f * scale);
    const int n_ticks   = (cfg.width - offset) / step_px;

    // X axis line
    DrawLine(offset, cfg.height - offset,
             cfg.width, cfg.height - offset, DARKGRAY);
    // Y axis line
    DrawLine(offset, cfg.height - offset,
             offset, 0, DARKGRAY);

    float range = cfg.val_max - cfg.val_min;

    // X ticks
    for (int t = 0; t <= n_ticks; ++t) {
        int px  = offset + t * step_px;
        float v = cfg.val_min + range / static_cast<float>(n_ticks) * t;
        DrawLine(px, cfg.height - offset - 3,
                 px, cfg.height - offset + 3, DARKGRAY);
        DrawText(TextFormat("%.0f", v),
                 px - 10, cfg.height - offset + 8, 9, DARKGRAY);
    }

    // Y ticks
    for (int t = 0; t <= n_ticks; ++t) {
        int py  = cfg.height - offset - t * step_px;
        float v = cfg.val_min + range / static_cast<float>(n_ticks) * t;
        DrawLine(offset - 3, py, offset + 3, py, DARKGRAY);
        DrawText(TextFormat("%.0f", v), 4, py - 5, 9, DARKGRAY);
    }
}

/// Draw a small legend box in the top-right corner.
static void draw_scatter_legend(const PlotConfig& cfg) {
    int lx = cfg.width - 155;
    int ly = 12;
    DrawRectangle(lx - 6, ly - 4, 150, 46, Fade(LIGHTGRAY, 0.6f));
    DrawRectangleLines(lx - 6, ly - 4, 150, 46, GRAY);

    DrawCircle(lx + 6,  ly + 6,  5, BLACK);
    DrawText("Data points",   lx + 16, ly + 1,  12, DARKGRAY);
    DrawCircle(lx + 6,  ly + 24, 5, RED);
    DrawText("Optimal point", lx + 16, ly + 19, 12, DARKGRAY);
}

static void draw_loss_legend(const PlotConfig& cfg) {
    int lx = cfg.width - 140;
    int ly = 12;
    DrawRectangle(lx - 6, ly - 4, 130, 28, Fade(LIGHTGRAY, 0.6f));
    DrawRectangleLines(lx - 6, ly - 4, 130, 28, GRAY);

    DrawLineEx({static_cast<float>(lx),     static_cast<float>(ly + 9)},
               {static_cast<float>(lx + 22), static_cast<float>(ly + 9)},
               2.0f, BLUE);
    DrawCircle(lx + 22, ly + 9, 3, DARKBLUE);
    DrawText("Loss", lx + 28, ly + 3, 12, DARKGRAY);
}

/// Map a data-space point to screen space.
static Vector2 data_to_screen(const Point& pt, const PlotConfig& cfg,
                               float scale, int offset)
{
    float sx = to_screen(pt.x, cfg.val_min, cfg.val_max,
                         cfg.width, offset, scale);
    float sy = to_screen(pt.y, cfg.val_min, cfg.val_max,
                         cfg.height, offset, scale);
    // Flip Y: data origin is bottom-left, screen origin is top-left
    sy = static_cast<float>(cfg.height) - sy;
    return {sx, sy};
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void plot_scatter(const std::vector<Point>& points, const PlotConfig& cfg) {
    InitWindow(cfg.width, cfg.height, cfg.title.c_str());
    SetTargetFPS(60);

    float scale = 1.0f;
    const int offset = 55;

    while (!WindowShouldClose()) {
        // Mouse-wheel zoom
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            scale += wheel * 0.1f;
            scale  = std::clamp(scale, 0.4f, 3.0f);
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        draw_axes(cfg, scale);

        // Draw points
        const int last = static_cast<int>(points.size()) - 1;
        for (int i = 0; i <= last; ++i) {
            Vector2 s  = data_to_screen(points[static_cast<std::size_t>(i)],
                                         cfg, scale, offset);
            bool is_last = (i == last && cfg.mark_last_red);
            Color col    = is_last ? RED : BLACK;
            float radius = is_last ? 7.0f : 4.5f;
            DrawCircleV(s, radius, col);
        }

        if (cfg.show_legend) draw_scatter_legend(cfg);

        // Title
        DrawText(cfg.title.c_str(), offset + 8, 10, 14, DARKGRAY);
        // Zoom hint
        DrawText("Scroll to zoom", cfg.width - 100, cfg.height - 20, 10, LIGHTGRAY);

        EndDrawing();
    }

    CloseWindow();
}

void plot_loss_curve(const std::vector<Point>& losses, const PlotConfig& cfg) {
    if (losses.empty()) return;

    InitWindow(cfg.width, cfg.height, cfg.title.c_str());
    SetTargetFPS(60);

    float scale = 1.0f;
    const int offset = 55;

    while (!WindowShouldClose()) {
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            scale += wheel * 0.1f;
            scale  = std::clamp(scale, 0.4f, 3.0f);
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        draw_axes(cfg, scale);

        // Draw connected line segments between consecutive loss points
        for (std::size_t i = 1; i < losses.size(); ++i) {
            Vector2 a = data_to_screen(losses[i - 1], cfg, scale, offset);
            Vector2 b = data_to_screen(losses[i],     cfg, scale, offset);
            DrawLineEx(a, b, 2.0f, BLUE);
            DrawCircleV(b, 3.0f, DARKBLUE);
        }
        // First point dot
        DrawCircleV(data_to_screen(losses[0], cfg, scale, offset), 3.0f, DARKBLUE);

        if (cfg.show_legend) draw_loss_legend(cfg);

        DrawText(cfg.title.c_str(), offset + 8, 10, 14, DARKGRAY);
        DrawText("Scroll to zoom", cfg.width - 100, cfg.height - 20, 10, LIGHTGRAY);

        EndDrawing();
    }

    CloseWindow();
}

void plot_loss_contour(const LossFn& loss,
                       const GradFn& grad,
                       const std::vector<Trajectory>& trajectories,
                       const PlotConfig& cfg)
{
    InitWindow(cfg.width, cfg.height, cfg.title.c_str());
    SetTargetFPS(60);

    constexpr int         kCell        = 8;    // heatmap cell size in pixels
    constexpr int         kLevels      = 12;   // number of contour bands
    constexpr int         kArrowStep   = 50;   // pixels between two gradient arrows
    constexpr float       kArrowLength = 20.0f;
    constexpr std::size_t kPaletteSize = 5;

    const int   cells_x = cfg.width  / kCell;
    const int   cells_y = cfg.height / kCell;
    const float range   = cfg.val_max - cfg.val_min;
    const float w       = static_cast<float>(cfg.width);
    const float h       = static_cast<float>(cfg.height);

    // The loss of every cell is evaluated ONCE here — doing it inside the
    // render loop would redo the expensive part 60 times per second.
    std::vector<float> grid(static_cast<std::size_t>(cells_x * cells_y));
    float lo = std::numeric_limits<float>::max();
    float hi = std::numeric_limits<float>::lowest();

    for (int i = 0; i < cells_y; ++i) {
        for (int j = 0; j < cells_x; ++j) {
            // Row 0 is the BOTTOM row (y = val_min); screen y goes down, flipped at draw time
            const Point p = {
                cfg.val_min + (static_cast<float>(j) + 0.5f) / static_cast<float>(cells_x) * range,
                cfg.val_min + (static_cast<float>(i) + 0.5f) / static_cast<float>(cells_y) * range
            };
            const float l = loss(p);
            grid[static_cast<std::size_t>(i * cells_x + j)] = l;
            lo = std::min(lo, l);
            hi = std::max(hi, l);
        }
    }

    std::vector<Color> colors(grid.size());
    for (std::size_t k = 0; k < grid.size(); ++k) {
        float t = (grid[k] - lo) / (hi - lo);        // 0 at the lowest loss, 1 at the highest
        t = std::sqrt(t);                            // loss grows fast far away; sqrt gives the area near the minimum more bands
        t = std::floor(t * static_cast<float>(kLevels)) / static_cast<float>(kLevels);  // snap to a band
        colors[k] = ColorFromHSV(240.0f * (1.0f - t), 0.6f, 0.95f);                     // blue -> red
    }

    const Color palette[kPaletteSize] = {BLACK, WHITE, MAGENTA, LIME, GOLD};

    // Data space -> screen space (y flipped)
    auto to_px = [&](const Point& pt) -> Vector2 {
        return { (pt.x - cfg.val_min) / range * w,
                 h - (pt.y - cfg.val_min) / range * h };
    };

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // Heatmap
        for (int i = 0; i < cells_y; ++i) {
            for (int j = 0; j < cells_x; ++j) {
                DrawRectangle(j * kCell, cfg.height - (i + 1) * kCell, kCell, kCell,
                              colors[static_cast<std::size_t>(i * cells_x + j)]);
            }
        }

        // Gradient field
        if (cfg.show_grad_vec || IsKeyDown(KEY_G)) {
            for (int py = kArrowStep / 2; py < cfg.height; py += kArrowStep) {
                for (int px = kArrowStep / 2; px < cfg.width; px += kArrowStep) {
                    const Point p = {
                        cfg.val_min + static_cast<float>(px) / w * range,
                        cfg.val_min + static_cast<float>(cfg.height - py) / h * range
                    };
                    const Point g = grad(p);
                    const float mag = std::sqrt(g.x * g.x + g.y * g.y);
                    if (mag < 1e-8f) continue;

                    const Vector2 from = {static_cast<float>(px), static_cast<float>(py)};
                    // Descent direction is -grad; screen y points down, so the y sign flips again
                    const Vector2 to   = {from.x + (-g.x / mag) * kArrowLength,
                                          from.y + ( g.y / mag) * kArrowLength};
                    DrawLineEx(from, to, 1.5f, BLACK);
                    DrawCircleV(to, 3.0f, BLACK);
                }
            }
        }

        // Trajectories
        for (std::size_t r = 0; r < trajectories.size(); ++r) {
            const Color col  = palette[r % kPaletteSize];
            const auto& path = trajectories[r].path;

            for (std::size_t k = 1; k < path.size(); ++k)
                DrawLineEx(to_px(path[k - 1]), to_px(path[k]), 1.5f, col);

            if (!path.empty())
                DrawCircleV(to_px(path.back()), 5.0f, col);   // where the optimizer ended up
        }

        // Legend
        const int legend_h = 22 * static_cast<int>(trajectories.size()) + 26;
        DrawRectangle(6, 6, 170, legend_h, Fade(LIGHTGRAY, 0.8f));
        DrawRectangleLines(6, 6, 170, legend_h, GRAY);
        for (std::size_t r = 0; r < trajectories.size(); ++r) {
            const int ly = 16 + 22 * static_cast<int>(r);
            DrawLineEx({14.0f, static_cast<float>(ly + 6)}, {40.0f, static_cast<float>(ly + 6)},
                       3.0f, palette[r % kPaletteSize]);
            DrawText(trajectories[r].name.c_str(), 48, ly, 14, DARKGRAY);
        }
        DrawText("hold G: gradient field", 14,
                 16 + 22 * static_cast<int>(trajectories.size()), 11, DARKGRAY);

        // Axis range hints
        DrawText(TextFormat("%.2f", static_cast<double>(cfg.val_min)), 4, cfg.height - 16, 11, DARKGRAY);
        DrawText(TextFormat("%.2f", static_cast<double>(cfg.val_max)), cfg.width - 40, cfg.height - 16, 11, DARKGRAY);
        DrawText(cfg.title.c_str(), cfg.width / 2 - 80, 10, 14, DARKGRAY);

        EndDrawing();
    }

    CloseWindow();
}

} // namespace viz
