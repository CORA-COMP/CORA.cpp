// isequal - whether a set is the empty set of the same dimension, as CORA's emptySet.isequal
//
// Syntax:   bool b = E.isequal(S);   bool b = E.isequal(S, tol);
// Inputs:   S - a set;  tol - nonnegative, default eps (CORA's); relaxes nothing here
// Outputs:  b - true if S is empty and has the dimension of E; false for another dimension, as
//           CORA (no error)
// See also: contains, Interval::isequal
//
// CORA: contSet/@emptySet/isequal.m

#include "cont_set/empty_set/private/priv_is_empty_set.h"
#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool EmptySet::isequal(const ContSet &S, double tol) const {
    detail::Entry entry;
    CORA_CHECK(tol >= 0, Error::wrong_value,
               [&] { return "isequal: tol must be nonnegative, got " + std::to_string(tol); });
    return S.dim() == n_ && priv_is_empty_set(S);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
