// isequal - whether a set is the whole space, as CORA's fullspace.isequal
//
// False for any other set, a point or another dimension. An empty interval is not the whole space
// (CORA says it is).
//
// Syntax:   bool b = F.isequal(S);   bool b = F.isequal(S, tol);
// Inputs:   S - a set (a point p: the interval [p, p])
//           tol - checked and otherwise unused; default: the repository tolerance
// Outputs:  b - true for a Fullspace or an Interval of that dimension with all bounds infinite
// See also: Interval::isequal
//
// CORA: contSet/@fullspace/isequal.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Fullspace::isequal(const ContSet &S, Tol tol) const {
    detail::Entry entry;
    static_cast<void>(resolve_tol(entry, tol, "isequal"));
    if (S.dim() != n_) return false;
    if (S.class_id() == ClassId::fullspace) return true;
    if (S.class_id() != ClassId::interval) return false;
    // infinite bounds must agree exactly, so only the box (-Inf, Inf)^n matches
    return static_cast<const Interval &>(S).isequal(Interval::unbounded(n_), 0.0);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
