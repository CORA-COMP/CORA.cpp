// test_cont_set_empty_set_fullspace - the registry pairs of the empty set and the whole space
//
// CORA: contSet/@contSet/reorder.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// Whether the registry lists op for the two classes.
bool aux_listed(BinaryOp op, const char *left, const char *right) {
    for (const BinaryPair &p : binary_pairs())
        if (p.op == op && p.left == left && p.right == right) return true;
    return false;
}

void the_pairs_are_listed_with_the_empty_set_first() {
    for (const BinaryOp op : {BinaryOp::plus, BinaryOp::intersect, BinaryOp::convex_hull,
                              BinaryOp::contains, BinaryOp::is_intersecting})
        check(aux_listed(op, "EmptySet", "Fullspace"),
              std::string(binary_op_name(op)) + "(EmptySet, Fullspace) is listed");
    check(aux_listed(BinaryOp::contains, "Fullspace", "EmptySet"), "contains(Fullspace, EmptySet)");
    check(precedence(ClassId::empty_set) < precedence(ClassId::fullspace), "CORA's precedence");
}

/// The sum and the intersection are empty, the hull is the whole space, in either order.
void the_sets_combine_as_cora_does(const std::string &b) {
    const EmptySet E(3);
    const Fullspace F(3);
    check(plus(E, F).is<EmptySet>() && plus(F, E).is<EmptySet>(), b + ": the sum is empty");
    check(intersect(E, F).is<EmptySet>() && intersect(F, E).is<EmptySet>(),
          b + ": the intersection is empty");
    check(convex_hull(E, F).is<Fullspace>() && convex_hull(F, E).is<Fullspace>(),
          b + ": the hull is the whole space");
    check(plus(Fullspace(0), EmptySet(0)).is<EmptySet>(), b + ": R^0 + the empty set of R^0");
}

/// The empty set contains no whole space and meets nothing; the whole space contains it.
void the_predicates_are_those_of_cora(const std::string &b) {
    const EmptySet E(3);
    const Fullspace F(3);
    check(!contains(E, F) && contains(F, E), b + ": containment");
    check(!is_intersecting(E, F) && !is_intersecting(F, E), b + ": no intersection");
    check(!F.isequal(E) && !E.isequal(F), b + ": different sets");
}

/// Other dimensions are refused; with the checks off the pair is still found.
void checks_and_their_switch() {
    check(test::raises([] { plus(EmptySet(2), Fullspace(3)); }, Error::dimension_mismatch),
          "plus: dimensions");
    check(test::raises([] { contains(Fullspace(2), EmptySet(3)); }, Error::dimension_mismatch),
          "contains: dimensions");
    ChecksScope off(false);
    check(plus(EmptySet(2), Fullspace(2)).is<EmptySet>(), "the checks off");
}

} // namespace

int main() {
    the_pairs_are_listed_with_the_empty_set_first();
    test::for_each_backend(the_sets_combine_as_cora_does);
    test::for_each_backend(the_predicates_are_those_of_cora);
    checks_and_their_switch();
    return test::finish("cont_set empty_set_fullspace");
}
