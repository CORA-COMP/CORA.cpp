// test_polytope_zono_bundle - Polytope::zono_bundle: the zonotope bundle of a bounded polytope
//
// CORA: contSet/@polytope/zonoBundle.m
// PROPERTY: contSet/@polytope/zonoBundle.m
//
// Property: the bundle holds exactly the polytope: its intersection has the same points
//
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// One zonotope per row, each the strip of the polytope along it.
void bundles(const std::string &b) {
    const ZonoBundle bundle = test::square_h().zono_bundle();
    check(bundle.parallel_sets() == 4, b + ": one zonotope per row");
    check(bundle.contains(column({0.5, 0.5})) && !bundle.contains(column({1.1, 0.5})),
          b + ": the points");
    // the vertex form and a triangle
    check(test::square_v().zono_bundle().parallel_sets() == 4, b + ": vertices");
    check(test::triangle_h().zono_bundle().parallel_sets() == 3, b + ": the triangle");
    // an equality gives two zonotopes
    const Polytope flat(test::square_h().A(), test::square_h().b(), Tensor({{1.0, -1.0}}),
                        column({0.0}));
    check(flat.zono_bundle().parallel_sets() == 6, b + ": an equality counts twice");
}

/// The bundle holds the points of the polytope and no others.
void properties(const std::string &b) {
    Rng rng(4);
    bool same = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P = test::random_polytope(rng, n, n + 1);
        const ZonoBundle bundle = P.zono_bundle();
        for (int k = 0; k < 30; ++k) {
            const Tensor x = column(test::random_direction(rng, n));
            same = same && bundle.contains(x, "exact", 1e-7) == P.contains(x, "exact", 1e-7);
        }
    }
    check(same, b + ": the bundle holds the points of P");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        bundles(b);
        properties(b);
    });
    check(test::raises([] { Polytope::inf(2).zono_bundle(); }, Error::special_error),
          "zono_bundle: R^n");
    check(test::raises([] { Polytope(Tensor({{1.0, 0.0}}), column({1.0})).zono_bundle(); },
                       Error::special_error),
          "zono_bundle: an unbounded set");
    ChecksScope off(false);
    check(test::square_h().zono_bundle().parallel_sets() == 4,
          "zono_bundle: the same with the checks off");
    return test::finish("polytope zono_bundle");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
