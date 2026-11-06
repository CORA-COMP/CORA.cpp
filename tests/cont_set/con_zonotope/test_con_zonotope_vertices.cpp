// test_con_zonotope_vertices - ConZonotope::vertices: the vertices of a constrained zonotope
//
// CORA: contSet/@conZonotope/vertices_.m
// PROPERTY: contSet/@conZonotope/vertices_.m
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

/// Whether the polygon has the corner (x, y).
bool aux_has(const Polygon &polygon, double x, double y) {
    for (const Point &p : polygon)
        if (close(p[0], x, 1e-9) && close(p[1], y, 1e-9)) return true;
    return false;
}

/// Hand-computed: the six corners of the hexagon, the ends of a segment, the corners of a square.
void vertices_known(const std::string &b) {
    const Polygon hex = Set(test::hexagon())->vertices()[0];
    check(hex.size() == 6 && aux_has(hex, 1, 0) && aux_has(hex, 1, -1) && aux_has(hex, 0, -1) &&
              aux_has(hex, -1, 0) && aux_has(hex, -1, 1) && aux_has(hex, 0, 1),
          b + ": the hexagon");
    const ConZonotope segment(column({0.0, 0.0}), Tensor({{1.0, 1.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), column({0.0}));
    const Polygon ends = Set(segment)->vertices()[0];
    check(ends.size() == 2 && aux_has(ends, 0, 1) && aux_has(ends, 0, -1), b + ": a segment");
    const ConZonotope square(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(Set(square)->vertices()[0].size() == 4, b + ": without constraints: the zonotope's");
    check(Set(ConZonotope::empty(2))->vertices()[0].empty(), b + ": the empty set has none");
    const ConZonotope plane(column({0.0, 0.0, 0.0}),
                            Tensor({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}),
                            Tensor({{1.0, 1.0, 1.0}}), column({0.0}));
    check(Set(plane)->vertices_3d()[0].size() == 6, b + ": the cube cut by a plane: six corners");
}

/// Stated property: every sampled point is inside the polygon (left of every edge).
void vertices_property(const std::string &b) {
    Rng rng(74);
    const ConZonotope cz = test::random_con_zonotope(rng, 2, 4, 2);
    const Polygon polygon = Set(cz)->vertices()[0];
    const Tensor points = test::sample_points(cz, rng, 30);
    const std::vector<double> xy = points.data();
    bool inside = true;
    for (int64_t k = 0; k < 30; ++k)
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            const Point &p = polygon[i], &q = polygon[(i + 1) % polygon.size()];
            const double cross =
                (q[0] - p[0]) * (xy[30 + k] - p[1]) - (q[1] - p[1]) * (xy[k] - p[0]);
            inside = inside && cross >= -1e-9;
        }
    check(inside, b + ": the sampled points are inside the polygon");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        vertices_known(b);
        vertices_property(b);
    });
    ChecksScope off(false);
    check(Set(test::hexagon())->vertices()[0].size() == 6,
          "vertices: the same with the checks off");
    return test::finish("con_zonotope vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
