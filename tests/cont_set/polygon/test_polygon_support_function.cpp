// test_polygon_support_function - Polygon::support_function: the largest d'x over the pieces
//
// CORA: contSet/@polygon/supportFunc_.m
// PROPERTY: contSet/@polygon/supportFunc_.m
// CORA authors: Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polygon_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: the support value is the largest d'v over all vertices.
void against_vertices() {
    Rng rng(11);
    for (int trial = 0; trial < 10; ++trial) {
        const Polygon pgon = test::random_polygon(rng, 3);
        const std::vector<double> d = test::random_direction(rng, 2);
        double best = -INFINITY;
        for (const Polytope &P : pgon.pieces()) {
            const std::vector<double> v = P.V().data();
            for (std::size_t l = 0; 2 * l < v.size(); ++l)
                best = std::max(best, d[0] * v[2 * l] + d[1] * v[2 * l + 1]);
        }
        check(close(pgon.support_function(vec(d)).data()[0], best),
              "support_function: the best vertex");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    against_vertices();
    check(std::isinf(Polygon::empty().support_function(vec({1.0, 0.0})).data()[0]),
          "support_function: -Inf for no piece");
    check(test::raises([] { test::unit_square().support_function(vec({1.0, 0.0, 0.0})); },
                       Error::dimension_mismatch),
          "support_function: a direction of three entries is refused");
    {
        ChecksScope off(false);
        check(close(test::unit_square().support_function(vec({1.0, 1.0})).data()[0], 2.0),
              "support_function: the same with the checks off");
    }
    return test::finish("polygon support_function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
