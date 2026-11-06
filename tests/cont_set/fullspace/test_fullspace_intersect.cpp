// test_fullspace_intersect - fullspace intersect: the intersection with any set is that set
//
// CORA: contSet/@fullspace/and_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void intersect_returns_the_operand(const std::string &b) {
    const Fullspace F(2);
    const Zonotope Z(column({1.0, 2.0}), Tensor::from_data({1, 0, 0, 1}, {2, 2}));
    const Interval I(column({0.0, 1.0}), column({2.0, 3.0}));
    const Set SZ = F.intersect(Z), SI = F.intersect(I), SF = F.intersect(Fullspace(2));
    check(SZ.is<Zonotope>() && SZ.as<Zonotope>().center().data() == Z.center().data(),
          b + ": a zonotope");
    check(SI.is<Interval>() && SI.as<Interval>().sup().data() == I.sup().data(), b + ": a box");
    check(SF.is<Fullspace>() && SF->dim() == 2, b + ": the whole space");
    check(intersect(F, Z).is<Zonotope>() && intersect(I, F).is<Interval>(),
          b + ": through the registry, in either order");
    check((F & Set(I)).is<Interval>(), b + ": the operator");
}

/// A set of another dimension is refused; the checks off give the same answer elsewhere.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([&] { F.intersect(Fullspace(3)); }, Error::dimension_mismatch),
          "intersect: another dimension");
    ChecksScope off(false);
    check(F.intersect(Fullspace(2)).is<Fullspace>(), "intersect: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(intersect_returns_the_operand);
    checks_and_their_switch();
    return test::finish("fullspace intersect");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
