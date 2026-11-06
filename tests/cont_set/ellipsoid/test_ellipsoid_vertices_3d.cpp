// test_ellipsoid_vertices_3d - Ellipsoid vertices_3d: points on the surface of a 3D ellipsoid
//
// CORA: contSet/@ellipsoid/plot3D.m
// PROPERTY: contSet/@ellipsoid/plot3D.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "ellipsoid_testing.h"

#include <cmath>
#include <limits>
#include <numbers>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: the 200 points are on the boundary.
void vertices_3d_property(const std::string &b) {
    Rng rng(149);
    const Ellipsoid E = test::random_ellipsoid(rng, 3);
    const std::vector<Point3> V = E.vertices_3d()[0];
    std::vector<double> flat;
    for (const Point3 &p : V) flat.insert(flat.end(), {p[0], p[1], p[2]});
    const Tensor P = Tensor::from_data(flat, {int64_t(V.size()), 3}).matrix_transpose();
    bool boundary = V.size() == 200;
    for (const double x : test::norms_of(E, P)) boundary = boundary && std::abs(x - 1.0) < 1e-6;
    check(boundary, b + ": 200 points on the boundary");
}

/// The empty ellipsoid is refused; with the checks off a valid call works.
void checks_and_their_switch() {
    check(test::raises([] { Ellipsoid::empty(3).vertices_3d(); }, Error::empty_set),
          "vertices_3d: empty");
    ChecksScope off(false);
    check(Ellipsoid(Tensor::eye(3)).vertices_3d()[0].size() == 200,
          "vertices_3d: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { vertices_3d_property(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid vertices_3d");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
