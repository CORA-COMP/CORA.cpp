// test_polytope_reduce_over_domain - Polytope::reduce_over_domain: one halfspace for a box
//
// CORA: contSet/@polytope/reduceOverDomain.m
// PROPERTY: contSet/@polytope/reduceOverDomain.m
//
// Property: over the domain the outer halfspace contains the set and the inner one lies inside it
//
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The unit square over its own box: the rows have the box [-1, 1]^2 and the offsets [0, 1], so the
/// offset is [0, 1] - ([-1, 1] [0, 1] + [-1, 1] [0, 1]) = [-2, 3].
void reductions(const std::string &b) {
    const Interval dom(column({0.0, 0.0}), column({1.0, 1.0}));
    const Polytope outer = test::square_h().reduce_over_domain(dom);
    check(outer.A().shape() == Shape{1, 2} && close(outer.A(), {0.0, 0.0}, 0) &&
              close(outer.b(), {3.0}, 0),
          b + ": outer: the center of the rows and the largest offset");
    const Polytope inner = test::square_h().reduce_over_domain(dom, "inner");
    check(close(inner.b(), {-2.0}, 0), b + ": inner: the smallest offset");
}

/// Over the domain the outer halfspace contains the set, and the inner one lies inside it.
void properties(const std::string &b) {
    Rng rng(4);
    bool ok = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P = test::random_polytope(rng, n, n + 1);
        const Interval dom = P.interval();
        const Polytope outer = P.reduce_over_domain(dom),
                       inner = P.reduce_over_domain(dom, "inner");
        ok = ok && outer.contains(test::sample_points(P, rng, 15), "exact", 1e-8);
        // points of the box that the inner halfspace holds are points of P
        const Tensor y = test::sample_points(Polytope(dom), rng, 40);
        for (int64_t k = 0; k < y.shape()[1]; ++k) {
            const Tensor p = y.slice(-1, k, k + 1);
            ok = ok && (!inner.contains(p, "exact", 0.0) || P.contains(p, "exact", 1e-9));
        }
    }
    check(ok, b + ": outer contains the set over the box, inner lies inside it");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        reductions(b);
        properties(b);
    });
    const Interval dom(column({0.0, 0.0}), column({1.0, 1.0}));
    check(test::raises([&] { test::square_h().reduce_over_domain(dom, "middle"); },
                       Error::wrong_value),
          "reduce_over_domain: the type");
    check(test::raises([&] { test::square_v().reduce_over_domain(dom); }, Error::special_error),
          "reduce_over_domain: a vertex polytope");
    check(test::raises(
              [] { test::square_h().reduce_over_domain(Interval(column({0.0}), column({1.0}))); },
              Error::dimension_mismatch),
          "reduce_over_domain: the dimension of the domain");
    ChecksScope off(false);
    check(test::square_h().reduce_over_domain(dom).dim() == 2,
          "reduce_over_domain: the same with the checks off");
    return test::finish("polytope reduce_over_domain");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
