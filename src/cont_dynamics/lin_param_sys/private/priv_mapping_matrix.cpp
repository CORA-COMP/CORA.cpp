// priv_mapping_matrix - the series of e^{A r} and its curvature, as CORA's priv_mappingMatrix
//
// A constant set with one generator takes expmOneParam (all terms as matrix zonotopes, no
// interval powers); otherwise A r goes to expmMixed (constant) or expmIndMixed (varying). F sums
// the boxes [i^(-i/(i-1)) - i^(-1/(i-1)), 0] (A r)^i / i! for i = 2..taylor_terms, plus the
// remainder E.
//
// Syntax:   LinParamSeries s = priv_mapping_matrix(sys, r, taylor_terms, intermediate_terms);
//           IntervalMatrix F = priv_tie(s, taylor_terms);
// Inputs:   sys - the system
//           r - the step size
//           taylor_terms - the order of the series
//           intermediate_terms - the order of its matrix-zonotope part
// Outputs:  s - the series
//           F - the correction matrix
// See also: priv_input_solution, init_reach
//
// CORA: contDynamics/@linParamSys/private/priv_mappingMatrix.m
// CORA: contDynamics/@linParamSys/private/priv_tie.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/lin_param_sys/private/priv.h"
#include "matrix_set/interval_matrix/private/priv_factorial.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The box [lower, 0] times the interval matrix M (CORA's 1 x 1 intervalMatrix times M).
IntervalMatrix aux_times_box(double lower, const IntervalMatrix &M) {
    const Tensor lo = M.center().full_like({1, 1}, lower), hi = lo.zeros_like();
    return IntervalMatrix(M.interval().times(Interval(lo, hi)));
}

} // namespace


// ===========================================  MAIN  =========================================== //

LinParamSeries priv_mapping_matrix(const LinParamSys &sys, double r, int taylor_terms,
                                   int intermediate_terms) {
    const MatZonotope &A = sys.A();
    // one generator and constant parameters: the series of C + b G in b (Althoff, ACC 2012)
    if (A.num_generators() == 1 && sys.type() == ParamType::constant) {
        MatZonotope::ExpmOneParamResult e = A.expm_one_param(r, taylor_terms);
        return {std::move(e.enclosure), e.remainder, std::move(e.powers), {}, e.remainder};
    }
    // otherwise the mixed series of A r, with dependent terms for constant parameters only
    const MatZonotope Ar = A * r;
    MatZonotope::ExpmMixedResult e = sys.type() == ParamType::constant
                                         ? Ar.expm_mixed(r, intermediate_terms, taylor_terms)
                                         : Ar.expm_ind_mixed(intermediate_terms, taylor_terms);
    return {std::move(e.enclosure), e.enclosure_rest, std::move(e.powers), std::move(e.powers2),
            e.remainder};
}

IntervalMatrix priv_power(const LinParamSeries &series, int i) {
    const auto z = static_cast<int>(series.zono_powers.size());
    if (i <= z) return series.zono_powers[std::size_t(i - 1)].interval_matrix();
    return series.int_powers[std::size_t(i - z - 1)];
}

IntervalMatrix priv_tie(const LinParamSeries &series, int taylor_terms) {
    // the second order with the box [-1/4, 0], then the orders 3..taylor_terms
    IntervalMatrix sum = aux_times_box(-0.25, priv_power(series, 2)) * (1 / priv_factorial(2));
    for (int i = 3; i <= taylor_terms; ++i) {
        const double lower = std::pow(i, -double(i) / (i - 1)) - std::pow(i, -1.0 / (i - 1));
        sum = sum + aux_times_box(lower, priv_power(series, i)) * (1 / priv_factorial(i));
    }
    return sum + series.remainder;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
