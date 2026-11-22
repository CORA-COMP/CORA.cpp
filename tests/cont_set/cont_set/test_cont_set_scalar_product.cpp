// test_cont_set_scalar_product - s * S and S * s on every set class
//
// CORA.cpp only: CORA writes the scaling of a set as a linear map (or mtimes with a number).
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The scaled set agrees with the set mapped by the scaled identity, on the box and from both
/// sides.
// both sides of the scaling agree with the scaled identity
void scaled(const std::string &b, const std::string &name, const ContSet &S, const ContSet &left,
            const ContSet &right) {
    const Tensor M = Tensor::eye(S.dim()) * 2.0;
    const Interval want = S.linear_map(M)->interval(), got = left.interval();
    check(test::close(got.inf(), want.inf(), 1e-12) && test::close(got.sup(), want.sup(), 1e-12),
          b + ": 2 * " + name);
    const Interval other = right.interval();
    check(test::close(other.inf(), want.inf(), 1e-12) &&
              test::close(other.sup(), want.sup(), 1e-12),
          b + ": " + name + " * 2");
    // the next step of the same case
}

void every_class(const std::string &b) {
    const Zonotope Z(vec({1.0, -1.0}), Tensor({{1.0, 0.0}, {0.0, 0.5}}));
    scaled(b, "Zonotope", Z, 2.0 * Z, Z * 2.0);
    const Capsule C(vec({1.0, -1.0}), vec({1.0, 0.0}), 0.5);
    scaled(b, "Capsule", C, 2.0 * C, C * 2.0);
    const ConZonotope cz(Z);
    scaled(b, "ConZonotope", cz, 2.0 * cz, cz * 2.0);
    const Ellipsoid E(vec({1.0, -1.0}), Tensor::eye(2));
    scaled(b, "Ellipsoid", E, 2.0 * E, E * 2.0);
    const Polytope P(Interval(vec({0.0, 0.0}), vec({1.0, 2.0})));
    scaled(b, "Polytope", P, 2.0 * P, P * 2.0);
    const ZonoBundle bundle = ZonoBundle::from(Z);
    scaled(b, "ZonoBundle", bundle, 2.0 * bundle, bundle * 2.0);
    const PolyZonotope poly(Z);
    scaled(b, "PolyZonotope", poly, 2.0 * poly, poly * 2.0);
    check((2.0 * Fullspace::inf(2)).is<Fullspace>(), b + ": 2 * Fullspace");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { every_class(b); });
    return test::finish("scalar product");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
