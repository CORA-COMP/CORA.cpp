// project_on_hyperplane - the orthogonal projection onto a hyperplane, as CORA's contSet
//
// CORA takes the hyperplane as a set and normalizes its constraint; CORA.cpp takes the
// constraint.
//
// Syntax:   R = S.project_on_hyperplane(c, d);
// Inputs:   S - set of dimension n
//           c - normal of the hyperplane {x | c'x = d}, a nonzero column (..., n, 1)
//           d - its offset, (..., 1, 1)
// Outputs:  R - (I - c c') S + d c with c and d divided by the length of c
// See also: linear_map, minus
//
// CORA: contSet/@contSet/projectOnHyperplane.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::project_on_hyperplane(const Tensor &c, const Tensor &d) const {
    detail::Entry entry;
    CORA_CHECK(is_column(c, dim()), Error::dimension_mismatch, [&] {
        return "project_on_hyperplane: the normal must be a column of the set's dimension " +
               std::to_string(dim()) + ", got the shape " + shape_text(c.shape());
    });
    CORA_CHECK(d.numel() == 1 && d.ndim() >= 2, Error::wrong_value, [&] {
        return "project_on_hyperplane: the offset must be a number (1, 1), got the shape " +
               shape_text(d.shape());
    });
    const Tensor length = c.vector_norm(Tensor::Norm::two, -2, true);
    CORA_CHECK(length.min().item() > 0, Error::wrong_value, [] {
        return std::string("project_on_hyperplane: the normal must not be zero (no hyperplane)");
    });
    // the constraint with a unit normal, as normalizeConstraints
    const Tensor unit = c.divide(length), offset = d.divide(length);
    const Tensor A = unit.eye_like(dim()) - unit.matmul(unit.matrix_transpose());
    return linear_map_impl(A)->translate_impl(unit.multiply(offset));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
