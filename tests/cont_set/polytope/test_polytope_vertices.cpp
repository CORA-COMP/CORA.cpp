// test_polytope_vertices - Polytope::vertices: the corners of a polytope in the plane and in space
//
// CORA: contSet/@polytope/vertices_.m
// PROPERTY: contSet/@polytope/vertices_.m
//
// Property: the corners are the extreme points, counter-clockwise in the plane
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Matthias Althoff
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

/// The square and the triangle: four and three corners, counter-clockwise.
void planes(const std::string &b) {
    const std::vector<Polygon> square = test::square_h().vertices();
    check(square.size() == 1 && square[0].size() == 4, b + ": four corners");
    // counter-clockwise: the signed area is positive
    double area = 0.0;
    for (std::size_t i = 0; i < square[0].size(); ++i) {
        const Point &p = square[0][i], &q = square[0][(i + 1) % square[0].size()];
        area += p[0] * q[1] - q[0] * p[1];
    }
    check(close(area / 2.0, 1.0), b + ": counter-clockwise, area 1");
    check(test::triangle_h().vertices()[0].size() == 3 &&
              test::square_v().vertices()[0].size() == 4,
          b + ": the triangle and the vertex form");
    check(Polytope::empty(2).vertices().empty(), b + ": the empty set has none");
}

/// A cube has eight corners in space.
void solids(const std::string &b) {
    const Polytope cube(Interval(column({0.0, 0.0, 0.0}), column({1.0, 1.0, 1.0})));
    const std::vector<std::vector<Point3>> corners = cube.vertices_3d();
    check(corners.size() == 1 && corners[0].size() == 8, b + ": the cube: eight corners");
}

/// The corners of random polygons lie in the set, and their shoelace area is its volume.
void properties(const std::string &b) {
    Rng rng(4);
    bool inside = true, same_area = true;
    for (int k = 0; k < 8; ++k) {
        const Polytope P = test::random_polytope(rng, 2, 5);
        const Polygon corners = P.vertices()[0];
        // the shoelace sum over the counter-clockwise corners
        double area = 0.0;
        for (std::size_t i = 0; i < corners.size(); ++i) {
            const Point &p = corners[i], &q = corners[(i + 1) % corners.size()];
            inside = inside && P.contains(column({p[0], p[1]}), "exact", 1e-8);
            area += p[0] * q[1] - q[0] * p[1];
        }
        same_area = same_area && close(area / 2.0, P.volume(), 1e-8);
    }
    check(inside && same_area, b + ": corners in the set, area = volume");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        planes(b);
        solids(b);
        properties(b);
    });
    // an unbounded set has no vertex list; a batch of halfspaces needs a program per member
    check(test::raises([] { Polytope(Tensor({{1.0, 0.0}}), column({1.0})).vertices(); },
                       Error::special_error),
          "vertices: an unbounded set");
    check(test::raises([] { test::square_h().vertices_3d(); }, Error::not_supported),
          "vertices: three dimensions needed");
    ChecksScope off(false);
    check(test::square_h().vertices()[0].size() == 4, "vertices: the same with the checks off");
    return test::finish("polytope vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
