// is_intersecting - whether a level set may meet a set, as CORA's levelSet.isIntersecting_
//
// The type is "approx": the set is replaced by its box and the constraints are evaluated over
// it. A '<=' row needs inf f below 0 up to within_tol, a '<' row inf f < 0, an '==' row
// inf f <= 0 <= sup f (each up to within_tol); every row must pass. False is certain, true says
// nothing. CORA tests an '==' row of a multi-constraint set like a '<' row (inf f < 0), which
// misses a box that touches the level set at inf f = 0; here an '==' row is decided as stated
// (docs/cora-differences.md).
//
// Syntax:   res = ls.is_intersecting(S, tol);
// Inputs:   S - a set of the dimension of ls
//           tol - the tolerance
// Outputs:  res - false if the set and the level set are disjoint
// See also: contains, fun_han
//
// CORA: contSet/@levelSet/isIntersecting_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/level_set/level_set.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether some point of a box with the constraint values [lo, hi] can satisfy the comparison.
bool aux_row_may_hold(CompOp op, double lo, double hi, double tol) {
    switch (op) {
    case CompOp::le:
        return lo < 0 || within_tol(lo, 0.0, tol);
    case CompOp::lt:
        return lo < 0;
    case CompOp::eq:
        return (hi > 0 || within_tol(hi, 0.0, tol)) && (lo < 0 || within_tol(lo, 0.0, tol));
    }
    CORA_FAIL(Error::report_to_dev, "is_intersecting: an unknown comparison");
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool LevelSet::is_intersecting(const ContSet &S, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, S.interval().inf().dtype(), "is_intersecting");
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "is_intersecting: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    const Interval I = S.interval();
    if (I.is_empty()) return false;
    const Interval F = fun_han(I);
    const std::vector<double> lo = F.inf().data(), hi = F.sup().data();
    for (std::size_t i = 0; i < eq_.size(); ++i)
        if (!aux_row_may_hold(ops_[i], lo[i], hi[i], tol)) return false;
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
