// test_con_zonotope_unite - ConZonotope::unite: a set that encloses the union
//
// CORA: contSet/@conZonotope/or.m
// PROPERTY: contSet/@conZonotope/or.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"
#include "cora/cont_set/registry.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the union of two boxes is enclosed, and the result has no constraints.
void unite_known(const std::string &b) {
    const ConZonotope left(column({-2.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const ConZonotope right(column({2.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const ConZonotope u = left.unite(right);
    check(u.A().numel() == 0, b + ": no constraints");
    check(u.contains(left.interval()) && u.contains(right.interval()),
          b + ": both boxes are inside");
    check(u.contains(column({0.0, 0.0})), b + ": and the gap, the set is convex");
    check(left.unite(Interval(column({1.0, -1.0}), column({3.0, 1.0}))).contains(right.interval()),
          b + ": with a box");
    check(left.unite(column({5.0, 5.0})).contains(column({5.0, 5.0})), b + ": with a point");
    check(left.unite({right, left}).contains(right.interval()), b + ": with a list");
    check((Set(left) | Set(right))->dim() == 2, b + ": the operator");
}

/// Stated property: sampled points of the constrained sets are inside the union.
void unite_property(const std::string &b) {
    cora::Rng rng(48);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope p = test::random_con_zonotope(rng, 2, 4, 1);
        const ConZonotope q = test::random_con_zonotope(rng, 2, 3, 1);
        const ConZonotope u = p.unite(q);
        check(u.contains(test::sample_points(p, rng, 15)) &&
                  u.contains(test::sample_points(q, rng, 15)),
              b + ": both sets are inside");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        unite_known(b);
        unite_property(b);
    });
    const ConZonotope a(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { a.unite(a, "tedrake"); }, Error::not_supported), "unite: tedrake");
    check(test::raises([&] { a.unite(a, "linprog", 2); }, Error::not_supported), "unite: order");
    check(test::raises([&] { a.unite(a, "other"); }, Error::wrong_value), "unite: method");
    check(test::raises([&] { a.unite(ConZonotope::origin(3)); }, Error::dimension_mismatch),
          "unite: dimension");
    ChecksScope off(false);
    check(a.unite(a).dim() == 2, "unite: the same with the checks off");
    return test::finish("con_zonotope unite");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
