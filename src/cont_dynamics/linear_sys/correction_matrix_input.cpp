// correction_matrix_input - the matrix G of the input's curvature enlargement, as in CORA
//
// Between two time points the effect of a constant input u curves away from the chord: G u
// encloses that. G sums the Taylor terms A^(i-1) dt^i / i! for i >= 2, each weighted by where
// t^i - t is extremal on [0, dt], and adds the remainder of the series past taylor_terms times dt.
//
// Syntax:   Interval G = sys.correction_matrix_input(time_step, taylor_terms);
// Inputs:   time_step - step size
//           taylor_terms - order of the Taylor series
// Outputs:  G - interval matrix (n, n)
// See also: correction_matrix_state, reach
//
// CORA: contDynamics/@linearSys/taylorMatrices.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   03-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval LinearSys::correction_matrix_input(double time_step, int taylor_terms) const {
    detail::Entry entry;
    CORA_CHECK(time_step > 0, Error::wrong_value, [&] {
        return "correction matrix: time_step must be positive, got " + std::to_string(time_step);
    });
    CORA_CHECK(taylor_terms >= 1, Error::wrong_value, [&] {
        return "correction matrix: taylor_terms must be at least 1, got " +
               std::to_string(taylor_terms);
    });
    Tensor Ai = A(); // A^(i-1)
    Tensor Gneg = A().zeros_like(), Gpos = A().zeros_like();
    double dt_over_fac = time_step; // dt^i / i!, after the update in the loop

    for (int i = 2; i <= taylor_terms + 1; ++i) {
        dt_over_fac *= time_step / i;
        // The weight is negative: the negative part of A^(i-1) bounds G from above.
        const double weight = priv_extremal_weight(i, dt_over_fac);
        Gpos = Gpos + Ai.minimum(0.0) * weight;
        Gneg = Gneg + Ai.maximum(0.0) * weight;
        Ai = Ai.matmul(A());
    }
    const Interval E = expm_remainder(time_step, taylor_terms);
    return {Gneg + E.inf() * time_step, Gpos + E.sup() * time_step};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
