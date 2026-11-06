// constraints_satisfied - a sufficient test that all of a zonotope satisfies C x < d, as constrSat
//
// The bound: the upper bound of every row of the box around C*Z is below d. For a batch, true if
// it holds for every member.
//
// Syntax:   res = Z.constraints_satisfied(C, d);
// Inputs:   C - matrix (k, n)
//           d - column (k, 1)
// Outputs:  res - true if C x < d holds for every x in Z, by a bound; false says nothing
// See also: interval, linear_map
//
// CORA: contSet/@zonotope/constrSat.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Zonotope::constraints_satisfied(const Tensor &C, const Tensor &d) const {
    detail::Entry entry;
    CORA_CHECK(C.ndim() >= 2 && C.shape()[C.ndim() - 1] == dim(), Error::dimension_mismatch, [&] {
        return "constraints_satisfied: C must have " + std::to_string(dim()) +
               " columns (the dimension of the zonotope), got the shape " + shape_text(C.shape());
    });
    CORA_CHECK(is_column(d, C.shape()[C.ndim() - 2]), Error::dimension_mismatch, [&] {
        return "constraints_satisfied: d must be a column with one entry per row of C, got the "
               "shape " +
               shape_text(d.shape());
    });
    const Tensor below = linear_map(C).interval().sup().detach().less(d.detach());
    return below.min().item() != 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
