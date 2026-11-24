// combine_next - the formula with nested next operators merged
//
// Syntax:   res = phi.combine_next();
// Inputs:   none
// Outputs:  res - next applied to temporal & and | pushed inside, next of next summed (where the
// sum is exact) See also: stl.h
//
// CORA: specification/@stl/combineNext.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Stl Stl::combine_next() const {
    detail::Entry entry;
    const StlKind k = kind();
    if (k == StlKind::and_) return lhs().combine_next().and_(rhs().combine_next());
    if (k == StlKind::or_) return lhs().combine_next().or_(rhs().combine_next());
    if (k == StlKind::not_) return lhs().combine_next().not_();
    if (k != StlKind::next) return *this;
    // X[a] of an operand: pushed through a temporal & | and through ~, merged with a next
    const Stl in = lhs();
    const double a = from();
    if ((in.kind() == StlKind::and_ || in.kind() == StlKind::or_) && in.temporal()) {
        const Stl l = in.lhs().next(a).combine_next(), r = in.rhs().next(a).combine_next();
        return in.kind() == StlKind::and_ ? l.and_(r) : l.or_(r);
    }
    if (in.kind() == StlKind::not_) return in.lhs().next(a).combine_next().not_();
    if (in.kind() != StlKind::next) return *this;
    // keep the sum where exact: CORALean's addUp <= addDown (at nearest, fails for NaN only)
    const double b = in.from(), sum = a + b;
    if (a >= 0 && b >= 0 && sum <= sum) return in.lhs().next(sum).combine_next();
    return in.lhs().next(b).combine_next().next(a);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
