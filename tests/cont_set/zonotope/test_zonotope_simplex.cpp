// test_zonotope_simplex - Zonotope::simplex: a simplex that contains a zonotope
//
// CORA: contSet/@zonotope/simplex.m
// PROPERTY: contSet/@zonotope/simplex.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: the simplex has n + 1 facets and contains the corners of the zonotope, tightly
/// on at least one facet.
void simplex_property(const std::string &b) {
    Rng rng(81);
    for (int n = 1; n <= 4; ++n) {
        const Zonotope Z = Zonotope::generate_random(rng, n);
        const Polytope P = Z.simplex();
        check(P.A().shape() == Shape{n + 1, n}, b + ": n + 1 facets in " + std::to_string(n) + "D");
        check(P.contains(Z.rand_point(30, rng, "extreme")) && P.contains(Z.center()),
              b + ": the zonotope is inside");
        const Tensor sums = P.A().matmul(P.A().matrix_transpose());
        check(close(sums.slice(0, 0, 1).slice(1, 0, 1), {1.0}, 1e-12), b + ": unit normals");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(simplex_property);
    const Zonotope Z(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { Zonotope::stack({Z, Z}).simplex(); }, Error::not_supported),
          "simplex: a batch");
    ChecksScope off(false);
    check(Z.simplex().dim() == 1, "simplex: the same with the checks off");
    return test::finish("zonotope simplex");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
