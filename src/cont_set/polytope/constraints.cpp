// constraints - the halfspace form of a polytope, as CORA's polytope.constraints
//
// Syntax:   Polytope::Constraints C = P.constraints();   C.A, C.b, C.Ae, C.be
// Inputs:   P - polytope
// Outputs:  C - the held halfspaces if there are any. From the vertices: none (p = 0) is the row
//           0 <= -1; one vertex v the equalities I x = v; in 1D the bounds (or an equality for a
//           point); else the affine hull of the vertices as equalities and the facets of their
//           hull inside it as inequalities with unit normals. CORA's qhull lists a facet once per
//           triangle, here each is listed once, in the order of its first vertices. Not a batch
// See also: Polytope::box, Polytope::compact
//
// CORA: contSet/@polytope/constraints.m

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope::Constraints Polytope::constraints() const {
    detail::Entry entry;
    if (is_h_rep()) return {A_, b_, Ae_, be_};
    const int64_t n = dim();
    // no vertex: the empty set, which is the row 0 <= -1
    if (priv_has_no_vertices(*this)) {
        if (n == 0)
            return {priv_zeros(V_, 0, 0), priv_zeros(V_, 0, 1), priv_zeros(V_, 0, 0),
                    priv_zeros(V_, 0, 1)};
        return {priv_zeros(V_, 1, n), priv_zeros(V_, 1, 1) - 1.0, priv_zeros(V_, 0, n),
                priv_zeros(V_, 0, 1)};
    }
    return priv_tensors_h(V_, priv_constraints_of_points(V_, priv_host_points(V_), n));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
