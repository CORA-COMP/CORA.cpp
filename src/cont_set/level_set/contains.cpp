// contains - whether a level set contains points or a box, as CORA's levelSet.contains_
//
// Points: every constraint must hold, f < tol and f <= tol up to within_tol for '<' and '<=', and
// |f| <= tol for '==' (CORA compares with tol, not with 0). A box: the constraints are evaluated
// over it (interval arithmetic), so true is certain and false says nothing. A '<=' row needs
// sup f below 0 up to within_tol, a '<' row sup f < 0, an '==' row inf f and sup f both within
// tol of 0. CORA tests an '==' row of a multi-constraint set like a '<' row (sup f < 0), which
// accepts a box where f is nonzero, and throws noops for a single '==' set; here an '==' row is
// decided as stated (docs/cora-differences.md). The empty box is contained in every level set.
//
// Syntax:   res = ls.contains(points, tol);   res = ls.contains(I, tol);
// Inputs:   points - states as columns (n, N)
//           I - a box (n, 1)
//           tol - the tolerance
// Outputs:  res - true if all points (the box) satisfy every constraint
// See also: is_intersecting, fun_han
//
// CORA: contSet/@levelSet/contains_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/level_set/level_set.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the value v of a constraint with the comparison op holds at a point.
bool aux_point_holds(CompOp op, double v, double tol) {
    switch (op) {
    case CompOp::le:
        return v < tol || within_tol(v, tol);
    case CompOp::lt:
        return v < tol;
    case CompOp::eq:
        return std::abs(v) < tol || within_tol(std::abs(v), tol);
    }
    CORA_FAIL(Error::report_to_dev, "contains: an unknown comparison");
}

/// Whether the values [lo, hi] of a constraint over a box all satisfy the comparison.
bool aux_box_holds(CompOp op, double lo, double hi, double tol) {
    switch (op) {
    case CompOp::le:
        return hi < 0 || within_tol(hi, 0.0, tol);
    case CompOp::lt:
        return hi < 0;
    case CompOp::eq:
        return within_tol(lo, 0.0, tol) && within_tol(hi, 0.0, tol);
    }
    CORA_FAIL(Error::report_to_dev, "contains: an unknown comparison");
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool LevelSet::contains(const Tensor &points, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, points.dtype(), "contains");
    CORA_CHECK(points.ndim() == 2 && points.shape()[0] == dim(), Error::dimension_mismatch, [&] {
        return "contains: the points must be a matrix with " + std::to_string(dim()) + " rows";
    });
    // the values per constraint (rows) and point (columns), row-major
    const std::vector<double> v = fun_han(points).data();
    const auto cols = std::size_t(points.shape()[1]);
    for (std::size_t i = 0; i < eq_.size(); ++i)
        for (std::size_t j = 0; j < cols; ++j)
            if (!aux_point_holds(ops_[i], v[i * cols + j], tol)) return false;
    return true;
}

bool LevelSet::contains(const Interval &I, Tol tol_arg) const {
    detail::Entry entry;
    // the empty box has no columns, so only a box with points has a dimension to check
    const double tol = resolve_tol(entry, tol_arg, I.inf().dtype(), "contains");
    CORA_CHECK(I.is_empty() || (I.inf().ndim() == 2 && I.inf().shape()[0] == dim()),
               Error::dimension_mismatch, [&] {
                   return "contains: the box must be a column of " + std::to_string(dim()) +
                          " states";
               });
    if (I.is_empty()) return true;
    // the values of the constraints over the box, per row
    const Interval F = fun_han(I);
    const std::vector<double> lo = F.inf().data(), hi = F.sup().data();
    for (std::size_t i = 0; i < eq_.size(); ++i)
        if (!aux_box_holds(ops_[i], lo[i], hi[i], tol)) return false;
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
