// priv_normal_form - the conjunctive and disjunctive normal form by distribution
//
// Syntax:   Stl cnf = priv_normal_form(phi, StlKind::or_, StlKind::and_, StlNnf::compute);
// See also: priv_normal_form.h
//
// CORA.cpp only: the conjunctive and disjunctive normal form by distribution
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "specification/stl/private/priv_normal_form.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// outer(a, b) or inner(a, b), as the kinds say.
Stl aux_build(StlKind kind, const Stl &a, const Stl &b) {
    return kind == StlKind::and_ ? a.and_(b) : a.or_(b);
}

/// One pass (aux_rewrite): the formula and the number of distributions made.
std::pair<Stl, int64_t> aux_rewrite(const Stl &phi, StlKind outer, StlKind inner) {
    if (phi.kind() != outer && phi.kind() != inner) return {phi, 0};
    auto [lhs, c1] = aux_rewrite(phi.lhs(), outer, inner);
    auto [rhs, c2] = aux_rewrite(phi.rhs(), outer, inner);
    const int64_t cnt = c1 + c2;
    if (phi.kind() == inner) return {aux_build(inner, lhs, rhs), cnt};
    // both operands are inner nodes: the four combinations, left to right
    if (lhs.kind() == inner && rhs.kind() == inner) {
        const Stl l1 = lhs.lhs(), l2 = lhs.rhs(), r1 = rhs.lhs(), r2 = rhs.rhs();
        const Stl a = aux_build(inner, aux_build(outer, l1, r1), aux_build(outer, l1, r2));
        return {aux_build(inner, aux_build(inner, a, aux_build(outer, l2, r1)),
                          aux_build(outer, l2, r2)),
                cnt + 1};
    }
    if (lhs.kind() == inner)
        return {
            aux_build(inner, aux_build(outer, rhs, lhs.lhs()), aux_build(outer, rhs, lhs.rhs())),
            cnt + 1};
    // CORA looks at the kind of the original right operand here (the rewritten one agrees)
    if (rhs.kind() == inner && phi.rhs().kind() == inner)
        return {
            aux_build(inner, aux_build(outer, lhs, rhs.lhs()), aux_build(outer, lhs, rhs.rhs())),
            cnt + 1};
    return {aux_build(outer, lhs, rhs), cnt};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Stl priv_normal_form(const Stl &phi, StlKind outer, StlKind inner, StlNnf nnf) {
    Stl res = nnf == StlNnf::given ? phi : phi.negation_normal_form();
    // the passes until nothing is distributed any more
    while (true) {
        auto [next, cnt] = aux_rewrite(res, outer, inner);
        res = next;
        if (cnt == 0) return res;
    }
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
