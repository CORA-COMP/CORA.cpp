// priv_spec_rows - the halfspaces of a specification's polytope, for linear verify
//
// verify decides halfspace sets by their rows. A safe or unsafe specification holds a Polytope;
// its constraints A y <= b give one row each, and an equality Ae y = be the two rows Ae y <= be
// and -Ae y <= -be. A polytope given by vertices is converted by Polytope::constraints. Any other
// set, a custom specification or a polytope without constraints is refused.
//
// Syntax:   std::vector<SpecRow> rows = priv_spec_rows(spec);
// Inputs:   spec - a safe set or an unsafe set whose set is a Polytope
// Outputs:  rows - the halfspaces {y | a'y <= b}; their intersection is the polytope
// See also: verify, priv_verify_ra_zonotope, priv_verify_ra_support_function
//
// CORA: contDynamics/@linearSys/verify.m
// CORA authors: Tobias Ladner, Mark Wetzlinger, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify.h"
#include "cora/cont_set/polytope/polytope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Appends the rows of a matrix (m, n) with the offsets (m, 1), each multiplied by sign.
void aux_append(std::vector<SpecRow> &rows, const Tensor &A, const Tensor &b, double sign) {
    if (A.numel() == 0) return;
    const std::vector<double> a = A.data(), offsets = b.data();
    const std::size_t n = static_cast<std::size_t>(A.shape().back());
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        SpecRow row{{}, sign * offsets[i]};
        for (std::size_t j = 0; j < n; ++j) row.a.push_back(sign * a[i * n + j]);
        rows.push_back(std::move(row));
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<SpecRow> priv_spec_rows(const Specification &spec) {
    // a refusal is a failure here, not a check: verify has checked the rest of its input
    if (spec.type() == SpecType::custom)
        CORA_FAIL(Error::not_supported,
                  "verify: decides safe and unsafe sets, not custom specifications");
    const Set &S = spec.set();
    if (!S.is<Polytope>())
        CORA_FAIL(Error::not_supported,
                  "verify: the set of a specification must be a Polytope, got a " + S.class_name());
    // the held constraints, or those computed from the vertices
    const Polytope::Constraints C = S.as<Polytope>().constraints();
    std::vector<SpecRow> rows;
    aux_append(rows, C.A, C.b, 1.0);
    aux_append(rows, C.Ae, C.be, 1.0);
    aux_append(rows, C.Ae, C.be, -1.0);
    if (rows.empty())
        CORA_FAIL(Error::wrong_value, "verify: the polytope of a specification has no constraints");
    return rows;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
