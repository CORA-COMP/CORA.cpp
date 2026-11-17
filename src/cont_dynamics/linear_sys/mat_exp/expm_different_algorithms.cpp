// expm_different_algorithms - the matrix exponential by one of CORA's algorithms, by style
//
// A is square. Style 4 gives the enclosure of expm. Another style is refused (CORA answers []).
// Nothing where expm_taylor_horner_scaling has none.
//
// Syntax:   std::optional<Interval> E = mat_exp::expm_different_algorithms(A, N, style);
//           E = mat_exp::expm_different_algorithms(A, N, 2, scaling_exponent);
// Inputs:   N - the order
//           style - 0, 1, 2 as expm_taylor_horner_scaling, or 4 for expm at step 1
//           scaling_exponent - for style 2
// Outputs:  E - the enclosure of exp(M) over the set
// See also: expm_taylor_horner_scaling, expm
//
// CORA: matrixSet/@intervalMatrix/expmDifferentAlgorithms.m
// CORA authors: Ivan Brkan
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_expm_series.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::optional<Interval> mat_exp::expm_different_algorithms(const Interval &A, int64_t max_order,
                                                           int64_t style,
                                                           int64_t scaling_exponent) {
    detail::Entry entry;
    if (entry.check()) mat_exp::check_series_input(A, max_order, style, scaling_exponent, true);
    if (style == 4) return expm(A, 1.0, max_order).enclosure;
    return expm_taylor_horner_scaling(A, max_order, style, scaling_exponent);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
