// test_specification_safe_set - specification safe_set: one halfspace and polytopes
//
// CORA: specification/@specification/specification.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/specification/specification.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The box `[1, 3] x [0, 2]` as a zonotope.
Zonotope box() { return {Tensor({2.0, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})}; }

/// The halfspace {x | a'x <= b} as a safe set.
Specification halfspace(double a1, double a2, double offset) {
    return Specification::safe_set(Polytope(Tensor({{a1, a2}}), Tensor({{offset}})));
}

void safe_set(const std::string &b) {
    const Zonotope Z = box();
    check(halfspace(1.0, 0.0, 3.0).check(Z), b + ": x1 <= 3 touches: inside");
    check(!halfspace(1.0, 0.0, 2.9).check(Z), b + ": x1 <= 2.9 is left");
    check(halfspace(-1.0, 0.0, -1.0).check(Z), b + ": x1 >= 1 holds");
    check(!halfspace(-1.0, 0.0, -1.5).check(Z), b + ": x1 >= 1.5 is left");
    check(halfspace(1.0, 1.0, 5.0).check(Z), b + ": x1 + x2 <= 5 holds");
    check(!halfspace(1.0, 1.0, 4.9).check(Z), b + ": x1 + x2 <= 4.9 is left");
}

/// Several rows are a polytope: all must hold. Any set class with a support function is decided.
void safe_polytope(const std::string &b) {
    const Zonotope Z = box();
    const Tensor A({{1.0, 0.0}, {0.0, 1.0}});
    check(Specification::safe_set(Polytope(A, Tensor({{3.0}, {2.0}}))).check(Z),
          b + ": inside both");
    const Specification low_top = Specification::safe_set(Polytope(A, Tensor({{3.0}, {1.0}})));
    check(!low_top.check(Z), b + ": one of two fails");
    check(low_top.type() == SpecType::safe_set, b + ": the type");
    check(low_top.set().is<Polytope>(), b + ": the set is kept");
    // the same polytope decides an interval, a constrained zonotope and a bundle
    const Specification spec = Specification::safe_set(Polytope(A, Tensor({{3.0}, {2.0}})));
    check(spec.check(Interval(Tensor({1.0, 0.0}), Tensor({3.0, 2.0}))), b + ": a box");
    check(spec.check(ConZonotope(Z)), b + ": a constrained zonotope");
    check(!spec.check(ConZonotope(Zonotope(Tensor({2.5, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})))),
          b + ": a shifted one is left");
    check(spec.check(ZonoBundle({Z, Z})), b + ": a bundle");
    check(spec.check(Ellipsoid(Tensor({{0.25, 0.0}, {0.0, 0.25}}), Tensor({2.0, 1.0}))),
          b + ": an ellipsoid");
    // a safe set may be any set, such as a zonotope
    check(Specification::safe_set(Zonotope(Tensor({2.0, 1.0}), Tensor({{2.0, 0.0}, {0.0, 2.0}})))
              .check(Z),
          b + ": a zonotope as the safe set");
}

/// A set of another dimension is refused, with and without checks.
void errors(const std::string &b) {
    check(test::raises(
              [] { halfspace(1.0, 0.0, 3.0).check(Zonotope(Tensor({1.0}), Tensor({{1.0}}))); },
              Error::dimension_mismatch),
          b + ": a set of another dimension");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        safe_set(b);
        safe_polytope(b);
        errors(b);
    });
    return test::finish("specification safe_set");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
