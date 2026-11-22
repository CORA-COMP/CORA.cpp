// test_cont_set_operators - the operators of the set classes: a sum, a shift and a map in C++
//
// CORA: contSet/@contSet/plus.m
// CORA: contSet/@contSet/minus.m
// CORA: contSet/@contSet/mtimes.m
// CORA authors: Tobias Ladner, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The operators give what the member functions give, bit for bit, for every class.
void operators_are_the_members(const std::string &b) {
    const Tensor v = vec({1.0, -2.0});
    const Tensor M({{2.0, 1.0}, {0.0, 3.0}});
    const Zonotope Z(vec({0.5, 0.0}), Tensor({{1.0, 0.0}, {0.0, 0.5}}));
    const Interval I(vec({-1.0, 0.0}), vec({1.0, 2.0}));
    const auto same = [&](const Tensor &a, const Tensor &b2, const std::string &what) {
        check(test::close(a, b2, 0.0), b + ": " + what);
    };

    // Capsule: a point shift and the sums with other sets.
    const Capsule C(vec({0.0, 0.0}), vec({1.0, 0.0}), 0.5);
    same((C + v).center(), C.plus(v).center(), "capsule + point");
    same((v + C).center(), C.plus(v).center(), "point + capsule");
    same((C - v).center(), C.plus(v * -1.0).center(), "capsule - point");
    same((C + I).center(), C.plus(I).center(), "capsule + interval");
    same((C + Z).center(), C.plus(Z).center(), "capsule + zonotope");
    same((C + C).center(), C.plus(C).center(), "capsule + capsule");

    // Ellipsoid: a point shift, the linear map and the sum.
    const Ellipsoid E(vec({1.0, 2.0}), Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    same((E + v).center(), E.plus(v).center(), "ellipsoid + point");
    same((v + E).center(), E.plus(v).center(), "point + ellipsoid");
    same((E - v).center(), E.plus(v * -1.0).center(), "ellipsoid - point");
    same((M * E).center(), E.linear_map(M).center(), "matrix * ellipsoid");
    same((E + E).center(), E.plus(E).center(), "ellipsoid + ellipsoid");

    // Zonotope bundle: a point shift, the sums and the linear map.
    const ZonoBundle ZB({Z, Z});
    same((ZB + v).center(), ZB.plus(v).center(), "bundle + point");
    same((v + ZB).center(), ZB.plus(v).center(), "point + bundle");
    same((ZB - v).center(), ZB.plus(v * -1.0).center(), "bundle - point");
    same((ZB + Z).center(), ZB.plus(Z).center(), "bundle + zonotope");
    same((ZB + I).center(), ZB.plus(I).center(), "bundle + interval");
    same((ZB + ZB).center(), ZB.plus(ZB).center(), "bundle + bundle");
    same((M * ZB).center(), ZB.linear_map(M).center(), "matrix * bundle");

    // Polynomial zonotope: a point shift, the sums and the linear map.
    const PolyZonotope PZ(Z);
    same((PZ + v).center(), PZ.plus(v).center(), "poly zonotope + point");
    same((v + PZ).center(), PZ.plus(v).center(), "point + poly zonotope");
    same((PZ - v).center(), PZ.plus(v * -1.0).center(), "poly zonotope - point");
    same((PZ + Z).center(), PZ.plus(Z).center(), "poly zonotope + zonotope");
    same((PZ + I).center(), PZ.plus(I).center(), "poly zonotope + interval");
    same((PZ + PZ).center(), PZ.plus(PZ).center(), "poly zonotope + poly zonotope");
    same((M * PZ).center(), PZ.linear_map(M).center(), "matrix * poly zonotope");

    // Constrained zonotope: the sums and the linear map.
    const ConZonotope CZ(Z.center(), Z.generators());
    same((CZ + Z).center(), CZ.plus(Z).center(), "con zonotope + zonotope");
    same((Z + CZ).center(), CZ.plus(Z).center(), "zonotope + con zonotope");
    same((CZ + I).center(), CZ.plus(I).center(), "con zonotope + interval");
    same((I + CZ).center(), CZ.plus(I).center(), "interval + con zonotope");

    // a number from either side on an interval
    same((I - 1.5).infimum(), (I + -1.5).infimum(), "interval - number");
    same((1.5 - I).infimum(), I.uminus().plus(Interval(vec({1.5, 1.5}), vec({1.5, 1.5}))).infimum(),
         "number - interval");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { operators_are_the_members(b); });
    return test::finish("cont set operators");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
