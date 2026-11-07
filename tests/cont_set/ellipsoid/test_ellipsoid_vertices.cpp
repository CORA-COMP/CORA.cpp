// test_ellipsoid_vertices - Ellipsoid vertices: the outline of a planar ellipsoid
//
// CORA: contSet/@ellipsoid/vertices_.m
// PROPERTY: contSet/@ellipsoid/vertices_.m
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

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: every vertex is on the boundary and the outline is counter-clockwise.
void vertices_property(const std::string &b) {
    Rng rng(139);
    for (int k = 0; k < 3; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 2);
        const Polygon V = E.vertices()[0];
        check(V.size() == 256, b + ": 256 vertices");
        std::vector<double> flat;
        for (const Point &p : V) flat.insert(flat.end(), {p[0], p[1]});
        const Tensor P = Tensor::from_data(flat, {256, 2}).matrix_transpose();
        bool boundary = true;
        for (const double x : test::norms_of(E, P)) boundary = boundary && std::abs(x - 1.0) < 1e-6;
        check(boundary, b + ": on the boundary");
        // the shoelace sum is twice the area, positive for a counter-clockwise outline
        double area = 0;
        for (std::size_t i = 0; i < V.size(); ++i) {
            const Point &p = V[i], &q = V[(i + 1) % V.size()];
            area += p[0] * q[1] - q[0] * p[1];
        }
        check(area > 0 && std::abs(area / 2 - E.volume().item()) < 0.01 * E.volume().item(),
              b + ": counter-clockwise, area near pi sqrt(det Q)");
    }
}

/// The empty ellipsoid and another dimension are refused; with the checks off a valid call works.
void checks_and_their_switch() {
    check(test::raises([] { Ellipsoid::empty(2).vertices(); }, Error::empty_set),
          "vertices: empty");
    check(test::raises([] { Ellipsoid(Tensor::eye(3)).vertices(); }, Error::not_supported),
          "vertices: 3D set");
    ChecksScope off(false);
    check(Ellipsoid(Tensor::eye(2)).vertices()[0].size() == 256,
          "vertices: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { vertices_property(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
