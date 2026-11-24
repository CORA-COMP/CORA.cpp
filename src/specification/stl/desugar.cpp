// desugar - the formula over until and the logic only
//
// Syntax:   res = phi.desugar();
// Inputs:   none
// Outputs:  res - until, &, |, ~ and the comparisons: release, finally, globally and next rewritten
// See also: stl.h
//
// CORA: specification/@stl/desugar.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Stl Stl::desugar() const {
    detail::Entry entry;
    switch (kind()) {
    case StlKind::until:
        return lhs().desugar().until(rhs().desugar(), interval());
    // phi R psi = ~(~phi U ~psi)
    case StlKind::release:
        return lhs().desugar().not_().until(rhs().desugar().not_(), interval()).not_();
    // F phi = true U phi, G phi = ~(true U ~phi), X[a] phi = true U[a, a] phi
    case StlKind::finally:
        return Stl(true).until(lhs().desugar(), interval());
    case StlKind::globally:
        return Stl(true).until(lhs().desugar().not_(), interval()).not_();
    case StlKind::next:
        return Stl(true).until(lhs().desugar(), interval());
    case StlKind::and_:
        return lhs().desugar().and_(rhs().desugar());
    case StlKind::or_:
        return lhs().desugar().or_(rhs().desugar());
    case StlKind::not_:
        return lhs().desugar().not_();
    case StlKind::true_:
    case StlKind::false_:
    case StlKind::comparison:
        return *this;
    }
    CORA_FAIL(Error::report_to_dev, "desugar: unknown formula kind");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
