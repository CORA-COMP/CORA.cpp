// test_cont_set_registry - the table of binary operations: its pairs, names, precedence and
// registration
//
// CORA: contSet/@contSet/contSet.m
// CORA: contSet/@contSet/reorder.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;

namespace {

/// The built-in pairs are listed, lower precedence first; the names round-trip.
void the_table_lists_its_pairs() {
    bool found = false;
    for (const BinaryPair &p : binary_pairs())
        found = found || (p.op == BinaryOp::plus && p.left == "Zonotope" && p.right == "Interval");
    check(found, "plus(Zonotope, Interval) is listed");
    check(precedence(ClassId::zonotope) < precedence(ClassId::interval), "CORA's precedence");
    for (int op = 0; op < static_cast<int>(BinaryOp::count); ++op)
        check(binary_op(binary_op_name(static_cast<BinaryOp>(op))) == static_cast<BinaryOp>(op),
              std::string("the name ") + binary_op_name(static_cast<BinaryOp>(op)));
    check(test::raises([] { binary_op("minus"); }, Error::wrong_value), "an unknown name");
}

/// A pair registered at run time is used, and listed; the checks off reach it too.
void a_registered_pair_is_used() {
    register_binary(
        BinaryOp::intersect, "Zonotope", "UnregisteredSet",
        SetFunction([](const ContSet &S1, const ContSet &) { return Set(S1.interval()); }));
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    const test::UnregisteredSet U;
    check(intersect(Z, U).is<Interval>(), "the registered function");
    check(intersect(U, Z).is<Interval>(), "and the other way round (commutative)");
    ChecksScope off(false);
    check(intersect(Z, U).is<Interval>(), "the checks off");
}

} // namespace

int main() {
    the_table_lists_its_pairs();
    a_registered_pair_is_used();
    return test::finish("cont_set registry");
}
