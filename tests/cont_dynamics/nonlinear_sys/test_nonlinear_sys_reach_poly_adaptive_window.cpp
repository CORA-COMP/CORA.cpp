// test_nonlinear_sys_reach_poly_adaptive_window - poly-adaptive window and time-varying input
//
// A rotation x' = A x + u(t) with A dt = 2 pi, so that the summed series matrix applied to the
// input set is almost 0: the integral of e^{As} over the step vanishes. The input u(t) = e^{-A(dt -
// t)} (r, 0) stays in the box |u| <= r but is aligned with the rotation at every time, so the state
// after the step is x0 + dt (r, 0) and the state over the step t e^{A(t - dt)} (r, 0) + e^{At} x0
// (plus the integral of the constant center). A set that applies the summed matrix to the varying
// input (or to the segment [0, u] of the center) misses these points; the term by term sum of the
// boxes of the series (CORALean's polyWindowFloat and polyPointFloat) holds them.
//
// CORA: contDynamics/@linearSys/initReach_adaptive.m
// PROPERTY: contDynamics/@linearSys/initReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Rotation {
    double w = 2 * kPi; ///< the angular speed, A = [0 -w; w 0], dt = 1

    Tensor A() const { return Tensor({{0.0, -w}, {w, 0.0}}); }

    /// e^{A t} v.
    std::array<double, 2> expm(double t, std::array<double, 2> v) const {
        const double c = std::cos(w * t), s = std::sin(w * t);
        return {c * v[0] - s * v[1], s * v[0] + c * v[1]};
    }
};

bool in_box(const PolyZonotope &Z, const std::array<double, 2> &x, double tol) {
    const Interval I = Z.interval();
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    return x[0] >= lo[0] - tol && x[0] <= hi[0] + tol && x[1] >= lo[1] - tol && x[1] <= hi[1] + tol;
}

/// The state at t of x(0) = x0, u(s) = center + e^{-A(1 - s)} (r, 0), dt = 1:
/// e^{At} x0 + integral of e^{As} center + t e^{A(t - 1)} (r, 0).
std::array<double, 2> state(const Rotation &R, double t, std::array<double, 2> x0,
                            std::array<double, 2> center, double r) {
    const auto e = R.expm(t, x0), v = R.expm(t - 1.0, {r, 0.0});
    const double s = std::sin(R.w * t) / R.w, k = (1 - std::cos(R.w * t)) / R.w;
    return {e[0] + center[0] * s - center[1] * k + t * v[0],
            e[1] + center[0] * k + center[1] * s + t * v[1]};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const Rotation R;
        const std::optional<PolySeries> series = PolySeries::make(R.A(), 1.0, 6);
        check(series.has_value(), b + ": an admissible order and halvings exist");
        if (!series) return;
        const Tensor zero = Tensor({0.0, 0.0});
        const PolyZonotope X0(Zonotope(zero, Tensor({{0.01, 0.0}, {0.0, 0.01}})));
        const PolyZonotope Vs(Tensor({0.0, 0.0}), Tensor::zeros({2, 0}), Tensor::zeros({2, 0}), {},
                              {});
        for (const bool shifted : {false, true}) {
            // the input box has the radius r = 0.1 about the center
            const std::array<double, 2> center =
                shifted ? std::array<double, 2>{0.3, 0.2} : std::array<double, 2>{0.0, 0.0};
            const double r = 0.1;
            const Zonotope V(Tensor({center[0], center[1]}), Tensor({{r, 0.0}, {0.0, r}}));
            const auto [window, diff] = priv_poly_window(*series, V, X0, 1e-4);
            const PolyZonotope end = priv_poly_point(*series, V, X0, Vs, 1e-4);
            bool inside = true, ends = true, diffs = true;
            for (const double sx : {-1.0, 0.0, 1.0})
                for (const double sy : {-1.0, 0.0, 1.0}) {
                    const std::array<double, 2> x0{0.01 * sx, 0.01 * sy};
                    for (int k = 0; k <= 40; ++k) {
                        const double t = double(k) / 40;
                        const auto x = state(R, t, x0, center, r);
                        inside = inside && in_box(window, x, 1e-9);
                        // the difference to the start: the set holds x(t) - x(0)
                        diffs = diffs && in_box(diff, {x[0] - x0[0], x[1] - x0[1]}, 1e-9);
                    }
                    ends = ends && in_box(end, state(R, 1.0, x0, center, r), 1e-9);
                }
            const std::string what = shifted ? "a shifted center" : "a centered input";
            check(inside, b + ": the window holds the states over the step, " + what);
            check(diffs, b + ": the differences hold x(t) - x(0), " + what);
            check(ends, b + ": the end set holds the state at the end, " + what);
            // the end set is far wider than the summed series matrix would make it
            const std::vector<double> hi = end.interval().sup().data();
            check(hi[0] > 0.09, b + ": the end set is as wide as the input drives it, " + what);
        }
    });
    return test::finish("nonlinear_sys reach poly adaptive window");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
