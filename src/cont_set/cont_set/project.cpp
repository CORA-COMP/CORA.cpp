// project - the projection of a set onto some of its dimensions, as CORA's contSet.project
//
// Syntax:   P = S.project(dims);
// Inputs:   dims - the dimensions to keep, 0-based, each in 0..n-1
// Outputs:  P - the projected set (a Set, of the class of S for zonotopes and intervals)
// See also: vertices, Zonotope::project_impl
//
// CORA: contSet/@contSet/project.m
// CORA authors: Tobias Ladner
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::project(const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    // only a box is a matrix; a bundle's center would cost linear programs
    const bool matrix_set = id_ == ClassId::interval || id_ == ClassId::custom;
    CORA_CHECK(!matrix_set || center().shape().back() <= 1, Error::not_supported, [] {
        return std::string("project: needs a set of vectors, not an interval matrix");
    });
    CORA_CHECK(!dims.empty(), Error::wrong_value,
               [] { return std::string("project: name at least one dimension, e.g. {0, 1}"); });
    for (const int64_t d : dims)
        CORA_CHECK(d >= 0 && d < dim(), Error::wrong_value, [&] {
            return "project: the dimension " + std::to_string(d) + " is outside 0.." +
                   std::to_string(dim() - 1) + " (dimensions are 0-based)";
        });
    return project_impl(dims);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
