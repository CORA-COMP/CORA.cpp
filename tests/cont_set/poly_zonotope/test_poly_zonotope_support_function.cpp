// test_poly_zonotope_support_function - PolyZonotope support function: an upper bound
//
// CORA: contSet/@polyZonotope/supportFunc_.m
// PROPERTY: contSet/@polyZonotope/supportFunc_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// pZ = {(1, 0) + a (1, 1) + a b (0, 2) + (0, 1) t}: generators on a and a b, one independent.
PolyZonotope known() {
    return {column({1.0, 0.0}),
            Tensor::from_data({1, 0, 1, 2}, {2, 2}),
            Tensor::from_data({0, 1}, {2, 1}),
            {{1, 0}, {1, 1}},
            {1, 2}};
}

/// Stated property: the value bounds d'x from above for the points x of the set.
void bound_property(const std::string &b) {
    Rng rng(5);
    for (int k = 0; k < 5; ++k) {
        const PolyZonotope P = test::random_poly(rng, 3);
        const std::vector<double> d = test::random_direction(rng, 3);
        const double value = test::support(P, d);
        const std::vector<double> x = test::sample(P, rng, 30).data();
        for (std::size_t j = 0; j < 30; ++j)
            check(d[0] * x[j] + d[1] * x[30 + j] + d[2] * x[60 + j] <= value + 1e-9,
                  b + ": the bound holds");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const PolyZonotope P = known();
        check(close(test::support(P, {1.0, 0.0}), 2.0, 1e-12), b + ": 1 + |1|");
        bound_property(b);
    });

    check(test::raises([] { known().support_function(column({1.0})); }, Error::dimension_mismatch),
          "support_function: the direction");
    ChecksScope off(false);
    check(test::support(known(), {1.0, 0.0}) > 0, "support_function: valid with the checks off");
    return test::finish("poly_zonotope support_function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
