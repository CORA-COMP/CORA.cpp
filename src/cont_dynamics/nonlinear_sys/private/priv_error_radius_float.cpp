// priv_error_radius_float - the radius of the solution of the abstraction error, as CORALean's
//
// The error V enters x' = A x + v, v in V, from 0: the boxes [A^i dt^(i+1) / (i+1)!] of the orders
// 0 to eta applied to V and added in turn, then the remainder [-dt E, dt E] applied to V, E the
// bound of the tail of the series of e^{A dt} by its norm (CORALean's errorSolutionAdaptiveFloat
// without a static input). CORA evaluates the remainder with expm, which no enclosure bounds and
// the lean backend lacks. lin-adaptive reads only the radius of this set (its tuning), so only the
// generators are formed, and the box generators that are 0 are left out of the sums (adding 0
// leaves a sum as it is): the sum of |G| of each row as CORALean's sumUp, bit for bit.
//
// Syntax:   Vec r = priv_error_radius_float(A, dt, eta, V);
// Inputs:   A - the matrix (n, n)
//           dt - the step size
//           eta - the order
//           V - the error set
// Outputs:  r - the sum of the absolute generators of the solution, per coordinate
// See also: priv_lin_reach_adaptive_pass.cpp, PolySeries::input_solution
//
// CORA: contDynamics/@linearSys/errorSolution_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       16-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_float_scalars.h"
#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Vec priv_error_radius_float(const Tensor &A, double dt, int eta, const Zonotope &V) {
    const auto n = static_cast<std::size_t>(A.size(-1));
    const auto h = static_cast<std::size_t>(V.generators().size(-1));
    const Vec a = A.data(), c = V.center().data(), G = V.generators().data();
    // what a coordinate of V reaches: |c_j| + the row sum of |G| (priv_interval_map)
    Vec reach(n);
    for (std::size_t j = 0; j < n; ++j) {
        double sum = 0;
        for (std::size_t l = h; l-- > 0;) sum = std::abs(G[j * h + l]) + sum;
        reach[j] = std::abs(c[j]) + sum;
    }
    // the generators of [lo, hi] V by rows: those of mid V, then the residual's box
    std::vector<Vec> rows(n);
    const auto add = [&](const Vec &lo, const Vec &hi) {
        Vec C(n * n), R(n * n);
        for (std::size_t e = 0; e < n * n; ++e) {
            C[e] = (lo[e] + hi[e]) * 0.5;
            R[e] = std::max(std::abs(lo[e] - C[e]), std::abs(hi[e] - C[e]));
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t l = 0; l < h; ++l) {
                double dot = 0;
                for (std::size_t k = n; k-- > 0;)
                    dot = priv_float_mul(C[i * n + k], G[k * h + l]) + dot;
                rows[i].push_back(std::abs(dot));
            }
            double box = 0;
            for (std::size_t k = n; k-- > 0;) box = priv_float_mul(R[i * n + k], reach[k]) + box;
            rows[i].push_back(box); // the other box generators of the row are 0
        }
    };
    // the orders 0 to eta, the powers of A by left multiplication (inputTaylorSumFloat)
    Vec power = priv_float_dense_identity(n);
    for (int i = 0; i <= eta; ++i) {
        if (i > 0) power = priv_float_dense_matmul(a, power, n);
        const auto [lo, hi] = priv_float_taylor_coeff(dt, i + 1);
        const auto [inf, sup] = priv_float_dense_box_scale(lo, hi, power);
        add(inf, sup);
    }
    // the remainder of the series by the norm of A dt (intervalInputRemainder)
    const double E =
        priv_float_remainder_bound(priv_float_dense_norm(priv_float_dense_scale(dt, a), n), eta);
    const double w = dt * E;
    add(Vec(n * n, -w), Vec(n * n, w));
    // the sums of the rows, from the last entry (sumUp)
    Vec radius(n);
    for (std::size_t i = 0; i < n; ++i) radius[i] = priv_sum_up(rows[i]);
    return radius;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
