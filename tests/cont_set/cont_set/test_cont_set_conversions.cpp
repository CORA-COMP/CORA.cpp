// test_cont_set_conversions - every arrow of the conversion matrix is a constructor of the target
//
// The constructor forwards to the member of the source class, so both spellings give the same set;
// the arrows CORA defines by an algorithm that is not ported raise Error::not_supported.
//
// CORA.cpp only: CORA writes each arrow as a function named after the target, on the source.
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
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

/// Whether two sets have the same box.
// the boxes of two sets agree
bool same_box(const ContSet &A, const ContSet &B) {
    const Interval a = A.interval(), b = B.interval();
    return test::close(a.inf(), b.inf(), 1e-12) && test::close(a.sup(), b.sup(), 1e-12);
}

void arrows(const std::string &b) {
    const Zonotope Z(vec({1.0, -1.0}), Tensor({{1.0, 0.5}, {0.0, 1.0}}));
    const Interval I(vec({0.0, -1.0}), vec({2.0, 1.0}));
    const Capsule C(vec({0.0, 0.0}), vec({1.0, 0.0}), 0.5);
    const ConZonotope cz(Z);
    const Polytope P(I);
    // the next step of the same case
    const ZonoBundle bundle = ZonoBundle::from(Z);
    const Ellipsoid E(vec({0.0, 0.0}), Tensor::eye(2));
    check(same_box(Zonotope(I), I.zonotope()), b + ": Zonotope(Interval)");
    check(same_box(Zonotope(C), C.zonotope()), b + ": Zonotope(Capsule)");
    check(same_box(Zonotope(cz), cz.zonotope()), b + ": Zonotope(ConZonotope)");
    check(same_box(Zonotope(E), E.zonotope()), b + ": Zonotope(Ellipsoid)");
    check(same_box(Zonotope(P), P.zonotope()), b + ": Zonotope(Polytope)");
    check(same_box(Zonotope(bundle), bundle.zonotope()), b + ": Zonotope(ZonoBundle)");
    check(same_box(Zonotope(PolyZonotope(Z)), Z), b + ": Zonotope(PolyZonotope)");
    check(same_box(Polytope(Z), Z.polytope()), b + ": Polytope(Zonotope)");
    check(same_box(Polytope(cz), cz.polytope()), b + ": Polytope(ConZonotope)");
    check(same_box(Polytope(bundle), bundle.polytope()), b + ": Polytope(ZonoBundle)");
    check(same_box(Polytope(C, "outer"), C.polytope("outer")), b + ": Polytope(Capsule)");
    check(same_box(ConZonotope(P), P.con_zonotope()), b + ": ConZonotope(Polytope)");
    check(same_box(ConZonotope(bundle), bundle.con_zonotope()), b + ": ConZonotope(ZonoBundle)");
    check(same_box(ZonoBundle(cz), cz.zono_bundle()), b + ": ZonoBundle(ConZonotope)");
    check(same_box(ZonoBundle(P), P.zono_bundle()), b + ": ZonoBundle(Polytope)");
    // a capsule with a generator is no ellipsoid, and the enclosing one needs an SDP
    check(test::raises([&] { static_cast<void>(Ellipsoid(C, "outer")); }, Error::not_supported),
          b + ": Ellipsoid(Capsule) forwards to the member");
}

/// The arrows into PolyZonotope and out of it that CORA computes by an algorithm.
// the arrows that CORA computes by an algorithm
void not_ported() {
    const Zonotope Z(vec({1.0, -1.0}), Tensor({{1.0, 0.5}, {0.0, 1.0}}));
    const PolyZonotope poly(Z);
    const Capsule C(vec({0.0, 0.0}), vec({1.0, 0.0}), 0.5);
    check(test::raises([&] { static_cast<void>(PolyZonotope(ConZonotope(Z))); },
                       Error::not_supported),
          "PolyZonotope(ConZonotope)");
    check(test::raises([&] { static_cast<void>(PolyZonotope(C)); }, Error::not_supported),
          "PolyZonotope(Capsule)");
    check(test::raises([&] { static_cast<void>(PolyZonotope(Ellipsoid(Tensor::eye(2)))); },
                       // the next step of the same case
                       Error::not_supported),
          "PolyZonotope(Ellipsoid)");
    check(test::raises([&] { static_cast<void>(PolyZonotope(Polytope(Z))); }, Error::not_supported),
          "PolyZonotope(Polytope)");
    check(test::raises([&] { static_cast<void>(PolyZonotope(ZonoBundle::from(Z))); },
                       Error::not_supported),
          "PolyZonotope(ZonoBundle)");
    check(test::raises([&] { static_cast<void>(Polytope(poly)); }, Error::not_supported),
          "Polytope(PolyZonotope)");
    check(test::raises([&] { static_cast<void>(Ellipsoid(poly)); }, Error::not_supported),
          "Ellipsoid(PolyZonotope)");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(arrows);
    not_ported();
    return test::finish("conversions");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
