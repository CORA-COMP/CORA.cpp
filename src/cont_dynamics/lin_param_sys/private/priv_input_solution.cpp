// priv_input_solution - the input of one step, as CORA's priv_inputSolution and priv_inputTie
//
// V = B U is mapped by the series r V + sum (A r)^i r / (i+1)! V (matrix zonotopes, then
// interval matrices) plus the remainder E r times the box of V; the constant input
// B u_trans + c by the interval matrix of the same series. If B (U + u_trans) + c misses the
// origin, the curvature of the constant input (priv_inputTie, Liou's remainder) is added.
//
// Syntax:   LinParamInput in = priv_input_solution(sys, series, U, u_trans, r, eta, order);
// Inputs:   sys - the system
//           series - priv_mapping_matrix of the step
//           U - the centered input set (m)
//           u_trans - the constant input (m, 1)
//           r - the step size
//           eta - the Taylor order
//           order - the zonotope order of the input solution
// Outputs:  in - Rinput, Rtrans and inputCorr
// See also: priv_mapping_matrix, init_reach
//
// CORA: contDynamics/@linParamSys/private/priv_inputSolution.m
// CORA: contDynamics/@linParamSys/private/priv_inputTie.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/lin_param_sys/private/priv.h"
#include "cora/global/tensor/tolerance.h"
#include "matrix_set/interval_matrix/private/priv_factorial.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The origin test ---------------------------------------------------------------------------------

/// Whether the inputs V + v_trans = B (U + u_trans) + c hold the origin (CORA's originContained;
/// CORA's linParamSys test ignores c unless B = 0, as c is not in v_trans there).
bool aux_origin_contained(const Zonotope &V, const Tensor &v_trans) {
    const Zonotope full = V + v_trans;
    const Tensor origin = full.center().zeros_like();
    if (full.representsa("interval", machine_eps())) return full.interval().contains(origin);
    return full.contains(origin);
}

/// The interval matrix whose every entry is [-v, v].
IntervalMatrix aux_all(const Tensor &like, int64_t n, double v) {
    const Tensor ones = like.ones_like({n, n});
    return IntervalMatrix(Interval(ones * (-1.0), ones).times(like.full_like({1, 1}, v)));
}

// The curvature of the constant input -------------------------------------------------------------

/// CORA's priv_inputTie: the boxes [(i^(-i/(i-1)) - i^(-1/(i-1))) r, 0] (A r)^(i-1) / i! summed
/// for i = 2..eta, plus Liou's remainder of the series (all of R if it does not converge).
IntervalMatrix aux_input_tie(const LinParamSys &sys, const LinParamSeries &series, double r,
                             int taylor_terms) {
    const auto times_box = [](double lower, const IntervalMatrix &M) {
        const Tensor lo = M.center().full_like({1, 1}, lower);
        return IntervalMatrix(M.interval().times(Interval(lo, lo.zeros_like())));
    };
    IntervalMatrix sum = times_box(-0.25 * r, priv_power(series, 1)) * (1 / priv_factorial(2));
    for (int i = 3; i <= taylor_terms; ++i) {
        const double lower = (std::pow(i, -double(i) / (i - 1)) - std::pow(i, -1.0 / (i - 1))) * r;
        sum = sum + times_box(lower, priv_power(series, i - 1)) * (1 / priv_factorial(i));
    }
    // the remainder of M. L. Liou (1966), with one power less for the constant input
    const double norm_A = sys.A().norm(Tensor::Norm::inf).item();
    const double epsilon = norm_A * r / (taylor_terms + 2);
    const int64_t n = sys.dim();
    if (epsilon >= 1) {
        // CORA prints 'Taylor order not high enough' and goes on with interval(-Inf, Inf)
        warn("CORA:linParamSys", "priv_inputTie: Taylor order not high enough (norm(A) r / "
                                 "(taylor_terms + 2) >= 1); the curvature is unbounded");
        const double inf = std::numeric_limits<double>::infinity();
        return sum + aux_all(sum.center(), n, inf);
    }
    const double phi = std::pow(norm_A, taylor_terms + 1) * std::pow(r, taylor_terms + 1) /
                       (priv_factorial(taylor_terms + 1) * (1 - epsilon));
    return sum + aux_all(sum.center(), n, phi / norm_A);
}

} // namespace


// ===========================================  MAIN  =========================================== //

LinParamInput priv_input_solution(const LinParamSys &sys, const LinParamSeries &series,
                                  const Zonotope &U, const Tensor &u_trans, double r,
                                  int taylor_terms, int zonotope_order) {
    const int64_t n = sys.dim();
    const Tensor &c = sys.c();
    // the centered inputs V = B U and the constant input B u_trans + c (cora-differences.md)
    const Zonotope V =
        sys.B() ? U.linear_map(*sys.B()) : Zonotope(c.zeros_like(), c.zeros_like({n, 0}));
    const Tensor v_trans = (sys.B() ? sys.B()->matmul(u_trans) : c.zeros_like()) + c;
    Zonotope input_set = V * r;
    IntervalMatrix integral(c.eye_like(n) * r);
    // the series: the matrix-zonotope powers, then the interval powers
    const auto z = static_cast<int>(series.zono_powers.size());
    for (int i = 1; i <= z; ++i) {
        const MatZonotope term =
            series.zono_powers[std::size_t(i - 1)] * (r / priv_factorial(i + 1));
        input_set = input_set + term * V;
        integral = integral + term.interval_matrix();
    }
    for (int i = z + 1; i <= z + static_cast<int>(series.int_powers.size()); ++i) {
        const IntervalMatrix term =
            series.int_powers[std::size_t(i - z - 1)] * (r / priv_factorial(i + 1));
        input_set = input_set + term * V;
        integral = integral + term;
    }
    // the remainder times the box of V
    const Tensor bound = V.interval().abs().sup();
    const Zonotope V_abs = Interval(bound * (-1.0), bound).zonotope();
    input_set = input_set + (series.remainder * r) * V_abs;
    integral = integral + series.remainder * r;
    // the constant input, and the sets without zero generators
    const Zonotope point(v_trans, v_trans.zeros_like({n, 0}));
    const double eps = machine_eps();
    const Zonotope trans = (integral * Interval(v_trans, v_trans).zonotope()).compact("zeros", eps);
    input_set = input_set.compact("zeros", eps);
    Zonotope correction(c.zeros_like(), c.zeros_like({n, 0}));
    if (!aux_origin_contained(V, v_trans))
        correction = aux_input_tie(sys, series, r, taylor_terms) * point;
    return {(input_set + trans).reduce(zonotope_order), trans, correction};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
