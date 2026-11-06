// test_zono_bundle_vertices - ZonoBundle::vertices in two and three dimensions
//
// CORA: contSet/@zonoBundle/vertices_.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the polygon has the corner (x, y).
bool aux_has(const Polygon &polygon, double x, double y) {
    for (const Point &p : polygon)
        if (close(p[0], x, 1e-9) && close(p[1], y, 1e-9)) return true;
    return false;
}

/// The members [-2, 2] x [-1, 1] and [-1, 3] x [-1, 1] meet in [-1, 2] x [-1, 1]; in space two
/// cubes meet in a cube.
void vertices_known(const std::string &b) {
    const Zonotope A(column({0.0, 0.0}), Tensor({{2.0, 0.0}, {0.0, 1.0}}));
    const Zonotope B(column({1.0, 0.0}), Tensor({{2.0, 0.0}, {0.0, 1.0}}));
    const Polygon square = Set(ZonoBundle({A, B}))->vertices()[0];
    check(square.size() == 4 && aux_has(square, -1, -1) && aux_has(square, 2, -1) &&
              aux_has(square, 2, 1) && aux_has(square, -1, 1),
          b + ": the four corners of the box");
    const Tensor G = Tensor({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    const Zonotope C(column({0.0, 0.0, 0.0}), G), D(column({1.0, 1.0, 1.0}), G);
    check(Set(ZonoBundle({C, D}))->vertices_3d()[0].size() == 8,
          b + ": the eight corners of the cube");
    check(Set(ZonoBundle::empty(2))->vertices()[0].empty(), b + ": the empty set has no vertex");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(vertices_known);
    ChecksScope off(false);
    check(Set(test::two_zonotopes())->vertices()[0].size() >= 4,
          "vertices: the same with the checks off");
    return test::finish("zono_bundle vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
