// priv_strip_refusals - why the strips, the gain or the observer of intersect_strip are refused
//
// Syntax:   Refusal r = priv_strip_refusal(n, C, phi, y);   priv_gain_refusal(n, C, lambda);
//           priv_observer_refusal(n, aux);
// Inputs:   n - dimension of the set
//           C (k, n), phi and y (k, 1) - the strips
//           lambda (n, k) - the gain
//           aux - the matrices E, F, A, C of an observer
// Outputs:  r - the identifier and the text of what is wrong, or nothing
// See also: Zonotope::intersect_strip, ConZonotope::intersect_strip
//
// CORA: contSet/@zonotope/intersectStrip.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Refusal priv_strip_refusal(int64_t n, const Tensor &C, const Tensor &phi, const Tensor &y) {
    if (C.ndim() != 2 || C.shape()[1] != n)
        return std::pair{Error::dimension_mismatch, "C must be a matrix (k, " + std::to_string(n) +
                                                        "), got the shape " +
                                                        shape_text(C.shape())};
    const int64_t k = C.shape()[0];
    if (phi.ndim() != 2 || !is_column(phi, k))
        return std::pair{Error::dimension_mismatch,
                         "phi must be a column (" + std::to_string(k) +
                             ", 1), one width per strip, got the shape " + shape_text(phi.shape())};
    if (y.ndim() != 2 || !is_column(y, k))
        return std::pair{Error::dimension_mismatch,
                         "y must be a column (" + std::to_string(k) +
                             ", 1), one center per strip, got the shape " + shape_text(y.shape())};
    return std::nullopt;
}

Refusal priv_gain_refusal(int64_t n, const Tensor &C, const Tensor &lambda) {
    if (lambda.ndim() == 2 && lambda.shape()[0] == n && lambda.shape()[1] == C.shape()[0])
        return std::nullopt;
    return std::pair{Error::dimension_mismatch, "the gain must be a matrix (" + std::to_string(n) +
                                                    ", " + std::to_string(C.shape()[0]) +
                                                    "), got the shape " +
                                                    shape_text(lambda.shape())};
    // the observer matrices must be two-dimensional and fit the dimension
}

Refusal priv_observer_refusal(int64_t n, const Zonotope::StripObserver &aux) {
    const auto rows = [](const Tensor &T) { return T.ndim() == 2 ? T.shape()[0] : -1; };
    const auto cols = [](const Tensor &T) { return T.ndim() == 2 ? T.shape()[1] : -1; };
    if (rows(aux.A) == n && cols(aux.A) == n && rows(aux.E) == n && cols(aux.C) == n &&
        rows(aux.F) == rows(aux.C) && rows(aux.F) >= 0)
        return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     std::string("the observer needs A (n, n), E (n, .), C (p, n) and F (p, .)")};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
