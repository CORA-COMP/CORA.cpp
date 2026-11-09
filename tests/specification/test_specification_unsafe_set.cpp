// test_specification_unsafe_set - specification unsafe_set: a halfspace the set must not touch
//
// CORA: specification/@specification/specification.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/specification/specification.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The box `[1, 3] x [0, 2]` as a zonotope.
Zonotope box() { return {Tensor({2.0, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})}; }

/// The unsafe halfspace {x | a'x <= b}.
Specification halfspace(double a1, double a2, double offset) {
    return Specification::unsafe_set(Polytope(Tensor({{a1, a2}}), Tensor({{offset}})));
}

/// The unsafe halfspace must not touch the set.
void unsafe_set(const std::string &b) {
    const Zonotope Z = box();
    check(halfspace(1.0, 0.0, 0.5).check(Z), b + ": x1 <= 0.5 is clear");
    check(!halfspace(1.0, 0.0, 1.0).check(Z), b + ": touching counts");
    check(!halfspace(1.0, 0.0, 2.0).check(Z), b + ": overlapping");
    check(!halfspace(1.0, 0.0, 9.0).check(Z), b + ": containing");
    check(halfspace(0.0, -1.0, -2.5).check(Z), b + ": x2 >= 2.5 is clear");
    check(halfspace(1.0, 0.0, 0.5).type() == SpecType::unsafe_set, b + ": the type");
    // any set class the polytope can intersect
    check(halfspace(1.0, 0.0, 0.5).check(Interval(Tensor({1.0, 0.0}), Tensor({3.0, 2.0}))),
          b + ": a box");
    check(!halfspace(1.0, 0.0, 1.5).check(ConZonotope(Z)), b + ": a constrained zonotope");
    check(halfspace(1.0, 0.0, 0.5).check(ZonoBundle({Z, Z})), b + ": a bundle");
    // no intersection test between a polytope and an ellipsoid: refused, not guessed
    check(test::raises(
              [&] { halfspace(1.0, 0.0, 0.5).check(Ellipsoid(Tensor({{1.0, 0.0}, {0.0, 1.0}}))); },
              Error::noops),
          b + ": an ellipsoid has no intersection test");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { unsafe_set(b); });
    return test::finish("specification unsafe_set");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
