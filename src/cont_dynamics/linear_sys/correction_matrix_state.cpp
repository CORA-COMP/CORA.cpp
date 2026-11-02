// correction_matrix_state - the matrix F of the curvature enlargement, as CORA's taylorMatrices
//
// The trajectories between two time points curve away from the chord between them; F X_k
// encloses that. F sums the Taylor terms i >= 2 of e^{A dt}, each weighted by where t^i - t is
// extremal on [0, dt], and adds the remainder of the series past order taylor_terms.
//
// Syntax:   Interval F = sys.correction_matrix_state(time_step, taylor_terms);
// Inputs:   time_step - step size;  taylor_terms - order of the Taylor series
// Outputs:  F - interval matrix (..., n, n)
// See also: reach
//
// CORA: contDynamics/@linearSys/taylorMatrices.m

#include "cont_dynamics/linear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The elementwise bound W on the series past the last term: |e^{|A| dt} - series|.
Tensor aux_remainder(const Tensor &Aabs, double time_step, const Tensor &series) {
    return ((Aabs * time_step).expm() - series).abs();
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval LinearSys::correction_matrix_state(double time_step, int taylor_terms) const {
    detail::Entry entry;
    CORA_CHECK(time_step > 0, Error::wrong_value, [&] {
        return "correction matrix: time_step must be positive, got " + std::to_string(time_step);
    });
    CORA_CHECK(taylor_terms >= 1, Error::wrong_value, [&] {
        return "correction matrix: taylor_terms must be at least 1, got " +
               std::to_string(taylor_terms);
    });
    const Tensor Aabs = A_.abs();
    Tensor Ai = A_, Aiabs = Aabs; // A^i and |A|^i
    Tensor Fneg = A_.zeros_like(), Fpos = A_.zeros_like();
    Tensor series = A_.eye_like() + Aabs * time_step; // sum_i |A|^i dt^i / i!, up to i = 1
    double dt_over_fac = time_step;                   // dt^i / i!

    for (int i = 2; i <= taylor_terms; ++i) {
        Ai = Ai.matmul(A_);
        Aiabs = Aiabs.matmul(Aabs);
        dt_over_fac *= time_step / i;
        series = series + Aiabs * dt_over_fac;
        // The weight is negative: the negative part of A^i bounds F from above.
        const double weight = priv_extremal_weight(i, dt_over_fac);
        Fpos = Fpos + Ai.neg() * weight;
        Fneg = Fneg + Ai.pos() * weight;
    }
    const Tensor W = aux_remainder(Aabs, time_step, series);
    return {Fneg - W, Fpos + W};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
