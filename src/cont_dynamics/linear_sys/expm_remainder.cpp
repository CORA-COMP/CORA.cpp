// expm_remainder - the error of the truncated series of e^{A dt}, as CORA's priv_expmRemainder
//
// The series sum_{i=0}^{taylor_terms} |A|^i dt^i / i! is compared with e^{|A| dt}; the absolute
// value keeps the difference free of cancellation.
//
// Syntax:   Interval E = sys.expm_remainder(time_step, taylor_terms);
// Inputs:   time_step - step size, positive
//           taylor_terms - order of the series, at least 1
// Outputs:  E - the box [-W, W] around zero with W = |e^{|A| dt} - series|, a building block of
//           reach
// See also: correction_matrix_input, input_solution
//
// CORA: contDynamics/@linearSys/private/priv_expmRemainder.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval LinearSys::expm_remainder(double time_step, int taylor_terms) const {
    detail::Entry entry;
    CORA_CHECK(time_step > 0 && taylor_terms >= 1, Error::wrong_value, [&] {
        return "expm_remainder: needs time_step > 0 and taylor_terms >= 1, got " +
               std::to_string(time_step) + " and " + std::to_string(taylor_terms);
    });
    const Tensor Aabs = A_.abs();
    Tensor Ai = A_.eye_like(), series = A_.eye_like(); // |A|^i and the series up to i
    double dt_over_fac = 1.0;                          // dt^i / i!

    for (int i = 1; i <= taylor_terms; ++i) {
        Ai = Ai.matmul(Aabs);
        dt_over_fac *= time_step / i;
        series = series + Ai * dt_over_fac;
    }
    const Tensor W = ((Aabs * time_step).expm() - series).abs();
    return {W * -1.0, W};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
