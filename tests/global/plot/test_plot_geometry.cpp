// test_plot_geometry - the union of polygons, the thinning of polygons and the hull of points
// in space, which a plot computes once for every front end

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/plot/geometry.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

namespace {

double area(const Polygon &p) {
    double a = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const Point &u = p[i], &v = p[(i + 1) % p.size()];
        a += u[0] * v[1] - v[0] * u[1];
    }
    return a / 2;
}

double total_area(const std::vector<Polygon> &loops) {
    double a = 0;
    for (const Polygon &p : loops) a += area(p);
    return a;
}

Polygon box(double x0, double y0, double x1, double y1) {
    return {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
}

void overlapping_polygons_have_one_outline() {
    const std::vector<Polygon> u = union_polygons({box(0, 0, 2, 2), box(1, 1, 3, 3)});
    check(u.size() == 1 && u[0].size() == 8, "two overlapping squares: one loop of eight corners");
    check(close(total_area(u), 7), "its area is 4 + 4 - 1");
    check(area(u[0]) > 0, "an outer loop runs counter-clockwise");
    // The same polygons clockwise, and in the other order, give the same union.
    Polygon cw = box(1, 1, 3, 3);
    std::reverse(cw.begin(), cw.end());
    check(close(total_area(union_polygons({cw, box(0, 0, 2, 2)})), 7),
          "orientation and order do not matter");
}

void shared_edges_vanish() {
    const std::vector<Polygon> u =
        union_polygons({box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)});
    check(u.size() == 1 && u[0].size() == 4, "three squares in a row are one rectangle");
    check(close(total_area(u), 3), "of area three");
    const std::vector<Polygon> same = union_polygons({box(0, 0, 1, 1), box(0, 0, 1, 1)});
    check(same.size() == 1 && same[0].size() == 4, "equal polygons are one");
}

void holes_are_clockwise_loops() {
    // Four rectangles around a unit hole: the square ring.
    const std::vector<Polygon> ring =
        union_polygons({box(0, 0, 3, 1), box(0, 2, 3, 3), box(0, 1, 1, 2), box(2, 1, 3, 2)});
    check(ring.size() == 2, "a ring has an outer loop and a hole");
    check(close(total_area(ring), 8), "outer 9 minus hole 1");
    check((area(ring[0]) > 0) != (area(ring[1]) > 0), "one loop turns each way");
}

void separate_and_nested_polygons() {
    check(union_polygons({box(0, 0, 1, 1), box(5, 5, 6, 6)}).size() == 2, "disjoint: two loops");
    const std::vector<Polygon> nested = union_polygons({box(0, 0, 4, 4), box(1, 1, 2, 2)});
    check(nested.size() == 1 && close(total_area(nested), 16),
          "a polygon inside another is absorbed");
    const std::vector<Polygon> touching = union_polygons({box(0, 0, 1, 1), box(1, 1, 2, 2)});
    check(close(total_area(touching), 2), "squares that touch in a corner keep both");
    check(union_polygons({{{0, 0}, {1, 1}}, {{2, 2}}}).empty(), "points and segments have no area");
    check(union_polygons({box(0, 0, 1, 1)}).size() == 1, "one polygon is its own union");
}

// Octagons along a line, each overlapping the next with a flat top and bottom, are the case of a
// reach set: their union is a long loop whose top and bottom edges are straight.
void many_overlapping_polygons() {
    std::vector<Polygon> sets;
    for (int k = 0; k < 200; ++k) {
        Polygon p;
        for (int i = 0; i < 8; ++i) {
            const double phi = (i + 0.5) * 0.785398163;
            p.push_back({0.1 * k + 0.3 * std::cos(phi), 0.3 * std::sin(phi)});
        }
        sets.push_back(p);
    }
    const std::vector<Polygon> u = union_polygons(sets);
    check(u.size() == 1, "a chain of 200 overlapping octagons is one loop");
    check(u[0].size() <= 12, "with only the boundary's corners");
    check(close(area(u[0]), 11.285680059411922, 1e-9), "and the area of the union");
}

// Polygons of many corners that overlap a great deal, as the sets of a long reach plot do.
void polygons_of_many_corners() {
    std::vector<Polygon> sets;
    for (int k = 0; k < 300; ++k) {
        Polygon p;
        for (int i = 0; i < 120; ++i) {
            const double phi = 2 * 3.14159265358979 * i / 120;
            p.push_back({0.01 * k + 1.0 * std::cos(phi), 0.3 * std::sin(phi)});
        }
        sets.push_back(p);
    }
    const std::vector<Polygon> u = union_polygons(sets);
    check(u.size() == 1, "300 overlapping ellipses of 120 corners are one loop");
    check(close(total_area(u), 2.7364778, 1e-3), "of the ellipse and the strip it sweeps");
    check(u[0].size() < 5000, "with far fewer corners than the 36000 of the ellipses");
    // With no tolerance the corners are those of the polygons.
    const std::vector<Polygon> exact = union_polygons({box(0, 0, 2, 2), box(1, 1, 3, 3)}, 0);
    check(exact.size() == 1 && exact[0].size() == 8, "a tolerance of zero keeps all corners");
}

// A polygon that is not convex is cut into triangles; where they meet there is no edge.
void polygons_that_are_not_convex() {
    const Polygon ell = {{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}};
    const std::vector<Polygon> single = union_polygons({ell, box(5, 5, 6, 6)});
    check(single.size() == 2 && close(total_area(single), 4), "an L and a square apart");
    const std::vector<Polygon> filled = union_polygons({ell, box(1, 1, 2, 2)});
    check(filled.size() == 1 && filled[0].size() == 4 && close(total_area(filled), 4),
          "an L and the square in its corner are a square");
}

void thinning_removes_needless_vertices() {
    const Polygon line = {{0, 0}, {1, 0}, {2, 0.001}, {3, 0}, {4, 0}};
    check(thin_polygon(line, {0.01, 0.01}, false).size() == 2,
          "a nearly straight line keeps its ends");
    check(thin_polygon(line, {0.0001, 0.0001}, false).size() == 5,
          "a tighter tolerance keeps it all");
    const Polygon square = {{0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
    const Polygon thin = thin_polygon(square, {0.01, 0.01});
    check(thin.size() == 4 && close(area(thin), 4), "the corners of a square stay");
    check(thin_polygon({{0, 0}, {1, 0}, {0, 0.001}}, {1, 1}).size() == 3,
          "a closed polygon keeps three vertices");
    // The tolerance is per axis: a spread of 0.5 is visible on a short axis only.
    const Polygon bump = {{0, 0}, {1, 0.5}, {2.5, 0}, {1, -3}};
    check(thin_polygon(bump, {10, 10}).size() == 3 ||
              thin_polygon(bump, {10, 10}).size() == bump.size(),
          "never fewer than three");
    check(thin_polygon(bump, {1, 0.1}).size() == 4, "a short axis keeps what a long one drops");
}

std::vector<Point3> cube_corners() {
    std::vector<Point3> p;
    p.reserve(8);
    for (int k = 0; k < 8; ++k)
        p.push_back({double(k & 1), double((k >> 1) & 1), double((k >> 2) & 1)});
    return p;
}

double face_area(const std::vector<Point3> &f) {
    double sx = 0, sy = 0, sz = 0;
    for (std::size_t i = 0; i < f.size(); ++i) {
        const Point3 &a = f[i], &b = f[(i + 1) % f.size()];
        sx += a[1] * b[2] - a[2] * b[1];
        sy += a[2] * b[0] - a[0] * b[2];
        sz += a[0] * b[1] - a[1] * b[0];
    }
    return 0.5 * std::sqrt(sx * sx + sy * sy + sz * sz);
}

// The first three points of a face turn counter-clockwise seen from outside: the normal they
// give points away from the center of the polytope.
bool faces_outward(const std::vector<std::vector<Point3>> &faces, const Point3 &center) {
    for (const std::vector<Point3> &f : faces) {
        const Point3 u = {f[1][0] - f[0][0], f[1][1] - f[0][1], f[1][2] - f[0][2]};
        const Point3 v = {f[2][0] - f[0][0], f[2][1] - f[0][1], f[2][2] - f[0][2]};
        const Point3 n = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                          u[0] * v[1] - u[1] * v[0]};
        if (n[0] * (f[0][0] - center[0]) + n[1] * (f[0][1] - center[1]) +
                n[2] * (f[0][2] - center[2]) <=
            0)
            return false;
    }
    return true;
}

void the_hull_of_a_cube_is_six_squares() {
    std::vector<Point3> points = cube_corners();
    points.push_back({0.5, 0.5, 0.5}); // inside
    points.push_back({0.5, 0.5, 0});   // on a face
    const std::vector<std::vector<Point3>> faces = hull_faces(points);
    check(faces.size() == 6, "six faces");
    for (const auto &f : faces)
        check(f.size() == 4 && close(face_area(f), 1), "each a unit square");
    check(faces_outward(faces, {0.5, 0.5, 0.5}), "counter-clockwise seen from outside");
}

void the_hull_of_a_tetrahedron_and_flat_sets() {
    const std::vector<std::vector<Point3>> tet =
        hull_faces({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    check(tet.size() == 4 && tet[0].size() == 3, "a tetrahedron has four triangles");
    const std::vector<std::vector<Point3>> flat =
        hull_faces({{0, 0, 1}, {2, 0, 1}, {2, 2, 1}, {0, 2, 1}, {1, 1, 1}});
    check(flat.size() == 1 && flat[0].size() == 4 && close(face_area(flat[0]), 4),
          "a flat set is one face");
    const std::vector<std::vector<Point3>> line =
        hull_faces({{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {0.5, 0.5, 0.5}});
    check(line.size() == 1 && line[0].size() == 2, "a segment is a face of two points");
    check(hull_faces({{1, 2, 3}, {1, 2, 3}}).size() == 1 && hull_faces({{1, 2, 3}})[0].size() == 1,
          "a point is a face of one");
    check(hull_faces({}).empty(), "nothing has no faces");
}

// A random zonotope's hull obeys Euler's formula and holds all its vertices.
void the_hull_of_a_zonotope() {
    Rng rng(4);
    for (int m : {3, 5, 8}) {
        const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, m);
        const std::vector<Point3> corners = Z.vertices_3d()[0];
        const std::vector<std::vector<Point3>> faces = hull_faces(corners);
        std::size_t edges = 0;
        for (const auto &f : faces) edges += f.size();
        check(corners.size() - edges / 2 + faces.size() == 2,
              "Euler's formula for m = " + std::to_string(m));
        check(faces.size() == static_cast<std::size_t>(m) * static_cast<std::size_t>(m - 1),
              "m (m - 1) faces for generic generators");
        Point3 c = {0, 0, 0};
        for (const Point3 &p : corners)
            for (int k = 0; k < 3; ++k)
                c[static_cast<std::size_t>(k)] +=
                    p[static_cast<std::size_t>(k)] / double(corners.size());
        check(faces_outward(faces, c), "outward for m = " + std::to_string(m));
    }
}

} // namespace

int main() {
    overlapping_polygons_have_one_outline();
    shared_edges_vanish();
    holes_are_clockwise_loops();
    separate_and_nested_polygons();
    many_overlapping_polygons();
    polygons_of_many_corners();
    polygons_that_are_not_convex();
    thinning_removes_needless_vertices();
    the_hull_of_a_cube_is_six_squares();
    the_hull_of_a_tetrahedron_and_flat_sets();
    the_hull_of_a_zonotope();
    return test::finish("plot geometry");
}
