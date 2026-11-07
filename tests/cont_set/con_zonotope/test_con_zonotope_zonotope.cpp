// test_con_zonotope_zonotope - ConZonotope::zonotope: a zonotope around a set
//
// CORA: contSet/@conZonotope/zonotope.m
// PROPERTY: contSet/@conZonotope/zonotope.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Without constraints the zonotope itself; the hexagon is inside its zonotope by both methods.
void zonotope_known(const std::string &b) {
    const ConZonotope plain(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    check(close(plain.zonotope().center(), {1, 2}, 0) &&
              close(plain.zonotope().generators(), {1, 0, 0, 2}, 0),
          b + ": without constraints");
    // a point: the constraints fix both factors
    const ConZonotope point(column({1.0, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                            Tensor({{1.0, 0.0}, {0.0, 1.0}}), column({0.5, 0.25}));
    check(close(point.zonotope().center(), {1.5, 1.25}, 1e-9) &&
              point.zonotope().generators().shape().back() <= 2,
          b + ": a point");
}

/// Stated property: the zonotope contains the vertices and the sampled points, for both methods.
void zonotope_property(const std::string &b) {
    Rng rng(77);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, 2);
        const Tensor points = test::sample_points(cz, rng, 20);
        check(cz.zonotope().contains(points), b + ": nullSpace contains the points");
        check(cz.zonotope("reduce").contains(points), b + ": reduce contains the points");
    }
    const Tensor corners = Tensor::from_data({1, 1, 0, -1, -1, 0, 0, -1, -1, 0, 1, 1}, {2, 6});
    check(test::hexagon().zonotope().contains(corners), b + ": the corners of the hexagon");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        zonotope_known(b);
        zonotope_property(b);
    });
    check(test::raises([] { ConZonotope::empty(2).zonotope(); }, Error::empty_set),
          "zonotope: the empty set");
    check(test::raises([] { test::hexagon().zonotope("pca"); }, Error::wrong_value),
          "zonotope: a method");
    ChecksScope off(false);
    check(test::hexagon().zonotope().dim() == 2, "zonotope: the same with the checks off");
    return test::finish("con_zonotope zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
