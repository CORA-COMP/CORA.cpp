// test_stl_robustness - how robustly a trace satisfies a formula
//
// CORA: specification/@stl/robustness.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/global/settings.h"
#include "cora/specification/stl.h"
#include "stl_testing.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether a is b up to tol, relative to the larger of 1 and |b|.
bool near(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol * std::max(1.0, std::abs(b));
}

/// The trace (sin t, cos 3t) at the times 0, 0.1, ..., 2.
void aux_trace(Tensor &x, std::vector<double> &t) {
    std::vector<double> d;
    for (int i = 0; i <= 20; ++i) {
        t.push_back(0.1 * i);
        d.push_back(std::sin(0.1 * i));
        d.push_back(std::cos(0.3 * i));
    }
    x = Tensor::from_data(d, {21, 2});
}

/// The nearest point of {x | A x <= b} among the candidates of all subsets of at most n rows.
double aux_distance_reference(const std::vector<std::vector<double>> &A,
                              const std::vector<double> &b, const std::vector<double> &p) {
    const std::size_t m = b.size();
    double best = INFINITY;
    // the equality projection onto the rows of a subset (n = 2: one or two rows), kept if feasible
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = i; j < m; ++j) {
            std::vector<double> q = p;
            if (i == j) {
                const double s = (A[i][0] * p[0] + A[i][1] * p[1] - b[i]) /
                                 (A[i][0] * A[i][0] + A[i][1] * A[i][1]);
                q = {p[0] - s * A[i][0], p[1] - s * A[i][1]};
            } else {
                const double det = A[i][0] * A[j][1] - A[i][1] * A[j][0];
                if (std::abs(det) < 1e-12) continue;
                q = {(b[i] * A[j][1] - A[i][1] * b[j]) / det,
                     (A[i][0] * b[j] - b[i] * A[j][0]) / det};
            }
            bool ok = true;
            for (std::size_t k = 0; k < m; ++k)
                ok = ok && A[k][0] * q[0] + A[k][1] * q[1] <= b[k] + 1e-9;
            if (ok) best = std::min(best, std::hypot(q[0] - p[0], q[1] - p[1]));
        }
    return best;
}

/// The robustness of formulas on traces, by the values of CORA.
void run(const std::string &b) {
    const auto v = StlExpr::variables("x", 2);
    const StlExpr x1 = v[0], x2 = v[1];
    Tensor x;
    std::vector<double> t;
    aux_trace(x, t);
    const Stl p = x1 > 0.8, q = x2 > -0.5;

    // the values MATLAB gives on this trace (CORA v2026.1.0); quadprog's are exact to 1e-8 only
    check(near((x1 < 0.5).robustness(x, t), 0.50000000000000022), b + ": predicate");
    check(near(p.finally(StlInterval(0, 1)).robustness(x, t), 0.041470984807896683), b + ": F");
    check(near((x2 > -0.9).globally(StlInterval(0.5, 1.2)).robustness(x, t), -0.089992496600445171),
          b + ": G");
    check(near((x1 < 0.9).release(q, StlInterval(0, 1)).robustness(x, t), 0.90000000000000024),
          b + ": release");
    check(near(q.until(p, StlInterval(0.2, 1)).robustness(x, t), -0.15578231276230869),
          b + ": until");
    check(near((x1 > 0.3).next(0.3).robustness(x, t), -0.0044797933386601652), b + ": next");
    check(near((x1 + x2 > 0.0)
                   .globally(StlInterval(0, 0.3))
                   .finally(StlInterval(0, 1))
                   .robustness(x, t),
               0.70710678118654768),
          b + ": nested");
    check(near(((x1 < 0.5) | (x2 < -0.8).finally(StlInterval(0, 0.5))).robustness(x, t),
               0.50000000000000022),
          b + ": or");
    // a level set, and a union of safe sets turned into unsafe ones
    check(near((x1.power(2) < 0.7).globally(StlInterval(0, 1)).robustness(x, t),
               -0.0080734182735712201),
          b + ": level set");
    check(near(((x1 < 0.5) | (x2 < 0.0)).globally(StlInterval(0, 1)).robustness(x, t),
               0.020574461730487936, 1e-8),
          b + ": union");
    check(
        near((((x1 > 0.5) & (x2 > 0.0)) | (x1 < -0.2)).finally(StlInterval(0, 2)).robustness(x, t),
             0.47384763088901077, 1e-8),
        b + ": polytope distance");

    // a trace not sampled uniformly is interpolated
    const std::vector<double> tn = {0, 0.1, 0.25, 0.5, 0.8, 1.0, 1.3};
    std::vector<double> d;
    for (const double s : tn) d.push_back(s), d.push_back(std::sin(2 * s));
    check(near((x1 < 1.2).globally(StlInterval(0, 1)).robustness(Tensor::from_data(d, {7, 2}), tn),
               0.33333333333333348),
          b + ": interpolated");

    // property: the distance to a polytope of random halfspaces by the nearest-point program
    Rng rng(4);
    for (int trial = 0; trial < 30; ++trial) {
        double u[6];
        rng.uniform(u, 6, -1.0, 1.0);
        // x1 > a & x2 > c, reversed by the union: a point in the corner region has a distance
        const Stl phi =
            ((x1 * 1.0 + x2 * u[0] > u[1]) | (x1 * u[2] + x2 > u[3])).globally(StlInterval(0, 0));
        const std::vector<double> pt = {u[4], u[5]};
        const Tensor xs = Tensor::from_data(pt, {1, 2});
        const double val = phi.robustness(xs, {0.0});
        // the unsafe set: both fail, {x1 + u0 x2 <= u1, u2 x1 + x2 <= u3}
        const double dist = aux_distance_reference({{1, u[0]}, {u[2], 1}}, {u[1], u[3]}, pt);
        const bool in_unsafe = pt[0] + u[0] * pt[1] <= u[1] && u[2] * pt[0] + pt[1] <= u[3];
        if (!in_unsafe && std::isfinite(dist)) check(near(val, dist), b + ": nearest point");
    }

    // invalid input and the checks off
    check(test::raises([&] { p.robustness(x, {0.0, 1.0}); }, Error::wrong_value), b + ": sizes");
    check(test::raises([&] { Stl(true).robustness(x, t); }, Error::not_supported), b + ": no set");
    {
        ChecksScope off(false);
        check(near((x1 < 0.5).robustness(x, t), 0.50000000000000022), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_robustness");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
