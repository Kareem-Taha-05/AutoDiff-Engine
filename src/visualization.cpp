#include "visualization.hpp"
#include "raylib.h"

#include <cmath>
#include <algorithm>
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

} // namespace viz
