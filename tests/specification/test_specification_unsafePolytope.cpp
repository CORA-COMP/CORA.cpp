// test_specification_unsafePolytope - an unsafe set of several halfspaces, as CORA's unsafeSet polytope
//
// The expected answers are those of MATLAB CORA's check(specification(polytope(A, b), 'unsafeSet'),
// zonotope(c, G)): safe iff the zonotope and the polytope have no common point.

#include "contSet/zonotope/zonotope.h"
#include "specification/specification.h"
#include "testing.h"

#include <stdexcept>

using namespace cora;
using test::check;

namespace {

Halfspace hs(double a1, double a2, double b) { return {Tensor({a1, a2}), b}; }

/// The box [l1, u1] x [l2, u2] as four halfspaces.
std::vector<Halfspace> box(double l1, double l2, double u1, double u2) {
    return {hs(1, 0, u1), hs(0, 1, u2), hs(-1, 0, -l1), hs(0, -1, -l2)};
}

bool safe(const Zonotope &Z, const std::vector<Halfspace> &P) {
    return Specification::unsafeSet(P).check(Z);
}

void unsafePolytope(const std::string &b) {
    const Zonotope Z1(Tensor({2.0, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));  // [1, 3] x [0, 2]
    check(safe(Z1, box(0, 0, 0.5, 2)), b + ": disjoint box");
    check(!safe(Z1, box(2.5, 0.5, 5, 1)), b + ": overlapping box");
    check(!safe(Z1, box(3, 0, 5, 2)), b + ": touching counts");
    check(safe(Z1, box(3.1, 0, 5, 2)), b + ": just apart");
    check(!safe(Z1, box(-5, -5, 5, 5)), b + ": containing box");

    // the segment from (0, 0) to (2, 2): each halfspace alone meets it, together they may not
    const Zonotope Z2(Tensor({1.0, 1.0}), Tensor({{1.0}, {1.0}}));
    check(safe(Z2, {hs(-1, 0, -1.5), hs(0, 1, 0.5)}), b + ": apart, no separating halfspace");
    check(!safe(Z2, {hs(-1, 0, -0.5), hs(0, 1, 1.5)}), b + ": overlapping segment");

    // three generators
    const Zonotope Z3(Tensor({0.0, 0.0}), Tensor({{1.0, 0.5, 0.2}, {0.0, 1.0, -0.3}}));
    check(safe(Z3, {hs(-1, -1, -3.5), hs(1, 0, 5)}), b + ": separated by one halfspace");
    check(safe(Z3, {hs(-1, -1, -3.0), hs(1, 0, 5)}), b + ": separated by one halfspace, close");
    check(safe(Z3, {hs(-1, -1, -2), hs(1, -1, 0), hs(0, 1, 0.5)}), b + ": three halfspaces");

    // the type, the halfspaces, and the errors
    const Specification spec = Specification::unsafeSet(box(0, 0, 1, 1));
    check(spec.type() == SpecType::UnsafeSet && spec.halfspaces().size() == 4, b + ": stored");
    bool thrown = false;
    try {
        Specification::unsafeSet(std::vector<Halfspace>{});
    } catch (const std::invalid_argument &) {
        thrown = true;
    }
    check(thrown, b + ": no halfspace is an error");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { unsafePolytope(b); });
    return test::finish("specification unsafePolytope");
}
