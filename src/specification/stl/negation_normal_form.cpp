// negation_normal_form - the formula with negations only at the comparisons
//
// Syntax:   res = phi.negation_normal_form();
// Inputs:   none
// Outputs:  res - the formula where ~ is pushed to the leaves; a negated comparison is the
// comparison that fails See also: stl.h
//
// CORA: specification/@stl/negationNormalForm.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// aux_recursive(obj, neg): the formula, negated if neg, in negation normal form. A negated
/// comparison becomes the one that fails (CORA swaps the sides and keeps the strictness).
Stl aux_push(const Stl &phi, bool neg) {
    const auto both = [&](StlKind k, const Stl &a, const Stl &b) {
        return k == StlKind::and_ ? a.and_(b) : a.or_(b);
    };
    switch (phi.kind()) {
    case StlKind::and_:
        return both(neg ? StlKind::or_ : StlKind::and_, aux_push(phi.lhs(), neg),
                    aux_push(phi.rhs(), neg));
    case StlKind::or_:
        return both(neg ? StlKind::and_ : StlKind::or_, aux_push(phi.lhs(), neg),
                    aux_push(phi.rhs(), neg));
    case StlKind::not_:
        return aux_push(phi.lhs(), !neg);
    // the temporal operators swap with their duals under a negation
    case StlKind::finally:
        return neg ? aux_push(phi.lhs(), true).globally(phi.interval())
                   : aux_push(phi.lhs(), false).finally(phi.interval());
    case StlKind::globally:
        return neg ? aux_push(phi.lhs(), true).finally(phi.interval())
                   : aux_push(phi.lhs(), false).globally(phi.interval());
    // until and release are duals of each other
    case StlKind::until:
        return neg ? aux_push(phi.lhs(), true).release(aux_push(phi.rhs(), true), phi.interval())
                   : aux_push(phi.lhs(), false).until(aux_push(phi.rhs(), false), phi.interval());
    case StlKind::release:
        return neg ? aux_push(phi.lhs(), true).until(aux_push(phi.rhs(), true), phi.interval())
                   : aux_push(phi.lhs(), false).release(aux_push(phi.rhs(), false), phi.interval());
    case StlKind::next:
        return aux_push(phi.lhs(), neg).next(phi.from());
    case StlKind::comparison:
        return neg ? Stl::compare(stl_cmp_negate(phi.cmp()), phi.expr_lhs(), phi.expr_rhs()) : phi;
    case StlKind::true_:
        return Stl(!neg);
    case StlKind::false_:
        return Stl(neg);
    }
    CORA_FAIL(Error::report_to_dev, "negation_normal_form: unknown formula kind");
}

} // namespace


// ===========================================  MAIN  =========================================== //

Stl Stl::negation_normal_form() const {
    detail::Entry entry;
    return aux_push(*this, false);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
