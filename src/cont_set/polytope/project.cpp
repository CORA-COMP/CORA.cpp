// project - the projection of a polytope onto some dimensions, as CORA's polytope.project
//
// Vertices are selected (V(dims, :), also for repeated dimensions and in a batch); halfspaces are
// projected by Fourier-Motzkin elimination of the other coordinates, with the rows normalized and
// compacted after every step (CORA's default 'fourier_jones' needs a compiled extension, and for
// two dimensions of a bounded set it goes through the vertices; the result is the same set). A
// repeated dimension of a halfspace set is Error::not_supported. Not a batch for halfspaces.
//
// Syntax:   Pres = P.project(dims);
// Inputs:   dims - the dimensions to keep (0-based, in the order of the result)
// Outputs:  Pres - the projection; a vertex set stays a vertex set
// See also: Polytope::lift, Polytope::project_high_dim
//
// CORA: contSet/@polytope/project.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polytope::project(const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    CORA_CHECK(!dims.empty(), Error::wrong_value,
               [] { return std::string("project: name at least one dimension, e.g. {0, 1}"); });
    for (const int64_t d : dims)
        CORA_CHECK(d >= 0 && d < dim(), Error::wrong_value, [&] {
            return "project: the dimension " + std::to_string(d) + " is outside 0.." +
                   std::to_string(dim() - 1) + " (dimensions are 0-based)";
        });
    // all dimensions in order: the set itself
    std::vector<int64_t> all(static_cast<std::size_t>(dim()));
    std::iota(all.begin(), all.end(), int64_t{0});
    if (dims == all) return *this;
    if (is_v_rep()) return Polytope(V_.take(dims, -2));
    std::vector<int64_t> sorted = dims;
    std::sort(sorted.begin(), sorted.end());
    CORA_CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
               Error::not_supported, [] {
                   return std::string("project: a repeated dimension is not supported for "
                                      "halfspaces");
               });
    const std::optional<HostH> projected = priv_project_h(priv_host_h(constraints()), dims);
    if (!projected) return empty(int64_t(dims.size()));
    const Constraints C = priv_tensors_h(A_, *projected);
    return {C.A, C.b, C.Ae, C.be};
}

Set Polytope::project_impl(const std::vector<int64_t> &dims) const { return project(dims); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
