// representsa - whether an interval matrix is also a set of another type, as CORA's representsa
//
// Syntax:   bool res = A.representsa(type);   bool res = A.representsa(type, tol);
// Inputs:   type - a type of Interval::representsa;  tol - as there (default: the repository
//           tolerance)
// Outputs:  res - the answer of Interval::representsa for the entries stacked into one vector
//           (CORA's reshape(I, [], 1)), so every type of the interval is available, not only
//           "emptySet" and "origin". The converted set (CORA's second output) is not returned
// See also: contains, Interval::representsa
//
// CORA: matrixSet/@intervalMatrix/representsa.m

#include "cont_set/cont_set/private/priv_set_type.h"
#include "cora/global/within_tol.h"
#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "matrix_set/interval_matrix/private/priv_vec.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool IntervalMatrix::representsa(const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(priv_is_set_type(type), Error::wrong_value,
               [&] { return "representsa: '" + type + "' is not a type of set that CORA knows"; });
    CORA_CHECK(type != "polygon", Error::not_supported,
               [] { return std::string("representsa: the type 'polygon' is not supported"); });
    const double tol = resolve_tol(entry, tol_arg, int_.inf().dtype(), "representsa");
    return priv_vec(int_).representsa(type, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
