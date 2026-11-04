// priv_predicates - whether an ellipsoid is empty or a point, and the refusal of a batch
//
// Syntax:   bool empty = priv_is_empty(E);   bool point = priv_is_point(E);
//           detail::Refusal r = priv_unbatched_refusal(E, "is_bigger");
// Inputs:   E - an ellipsoid;  operation - the name for the message
// Outputs:  the answer, or the identifier and text of the refusal
// See also: priv.h, Ellipsoid::contains
//
// CORA: contSet/@ellipsoid/representsa_.m

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_is_empty(const Ellipsoid &E) { return E.Q().numel() == 0; }

bool priv_is_point(const Ellipsoid &E) {
    if (priv_is_empty(E)) return false;
    // CORA's representsa_(E, 'point', eps): every entry of Q within eps of 0
    const Tensor near_zero = within_tol(E.Q(), E.Q().zeros_like(), machine_eps(E.Q().dtype()));
    return near_zero.reshape({near_zero.numel()}).all(0).item() != 0.0;
}

detail::Refusal priv_unbatched_refusal(const Ellipsoid &E, const char *operation) {
    if (E.Q().ndim() == 2) return std::nullopt;
    return std::pair{Error::not_supported,
                     std::string(operation) +
                         ": it decides one ellipsoid; a batch is not supported"};
}

detail::Refusal priv_pair_refusal(const Ellipsoid &E1, const Ellipsoid &E2, const char *operation) {
    const std::string name = operation;
    if (E1.dim() != E2.dim())
        return std::pair{Error::dimension_mismatch,
                         name + ": the ellipsoids differ in dimension, " +
                             std::to_string(E1.dim()) + " and " + std::to_string(E2.dim())};
    detail::Refusal refusal = priv_unbatched_refusal(E1, operation);
    if (!refusal) refusal = priv_unbatched_refusal(E2, operation);
    if (!refusal && (priv_is_empty(E1) || priv_is_empty(E2)))
        refusal = std::pair{Error::empty_set, name + ": an ellipsoid is empty"};
    return refusal;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
