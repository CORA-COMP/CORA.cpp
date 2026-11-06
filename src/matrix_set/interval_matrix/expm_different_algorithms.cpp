// expm_different_algorithms - the matrix exponential by one of CORA's algorithms, by style
//
// A is square. Style 4 gives the enclosure of expm. Another style is refused (CORA answers []).
// Nothing where expm_taylor_horner_scaling has none.
//
// Syntax:   std::optional<IntervalMatrix> E = A.expm_different_algorithms(N, style);
//           E = A.expm_different_algorithms(N, 2, scaling_exponent);
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
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::optional<IntervalMatrix>
IntervalMatrix::expm_different_algorithms(int64_t max_order, int64_t style,
                                          int64_t scaling_exponent) const {
    detail::Entry entry;
    if (entry.check()) check_series_input(max_order, style, scaling_exponent, true);
    if (style == 4) return expm(1.0, max_order).enclosure;
    return expm_taylor_horner_scaling(max_order, style, scaling_exponent);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
