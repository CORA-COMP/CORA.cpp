// test_cont_set_point_masks - the point tests of three classes answer one flag per point
//
// CORA.cpp only: CORA's predicates answer one logical for all points; the mask is CORA.cpp's.
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Three points: inside, on the boundary side, outside.
Tensor points() { return Tensor({{0.0, 0.0}, {0.5, 0.5}, {3.0, 3.0}}); }

/// Whether asking a Mask of several members for its bool throws, as NumPy's truth value does.
// asking for the bool of several flags is refused
bool bool_throws(const Mask &m) {
    try {
        static_cast<void>(static_cast<bool>(m));
    } catch (const Error &) {
        return true;
    }
    return false;
}

void zonotope_points(const std::string &b) {
    const Zonotope Z(vec({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    // the next step of the same case
    const Mask in = Z.contains(points());
    check(test::close(in.tensor(), {1.0, 1.0, 0.0}), b + ": Zonotope contains, a flag per point");
    check(!in.all() && in.any() && bool_throws(in), b + ": all, any and the bool of three flags");
    check(test::close(Z.is_intersecting(points()).tensor(), {1.0, 1.0, 0.0}),
          b + ": Zonotope is_intersecting");
    const Mask one = Z.contains(vec({0.5, 0.5}));
    check(one.tensor().ndim() == 0 && static_cast<bool>(one), b + ": one point, a 0-d mask");
    check(test::close(Z.contains(points().reshape({3, 1, 2})).tensor().reshape({3, 1}),
                      {1.0, 1.0, 0.0}),
          b + ": the shape of the points without the last axis");
    // the next step of the same case
}

void ellipsoid_points(const std::string &b) {
    const Ellipsoid E(vec({0.0, 0.0}), Tensor::eye(2));
    check(test::close(E.contains(points()).tensor(), {1.0, 1.0, 0.0}),
          b + ": Ellipsoid contains, a flag per point");
    check(test::close(E.is_intersecting(points()).tensor(), {1.0, 1.0, 0.0}),
          b + ": Ellipsoid is_intersecting");
    check(test::close(Ellipsoid::empty(2).contains(points()).tensor(), {0.0, 0.0, 0.0}),
          b + ": the empty ellipsoid contains none");
}

void capsule_points(const std::string &b) {
    const Capsule C(vec({0.0, 0.0}), vec({1.0, 0.0}), 1.0);
    check(test::close(C.contains(points()).tensor(), {1.0, 1.0, 0.0}),
          b + ": Capsule contains, a flag per point");
    check(test::close(C.is_intersecting(points()).tensor(), {1.0, 1.0, 0.0}),
          b + ": Capsule is_intersecting");
    check(static_cast<bool>(C.contains(vec({0.5, 0.5}))), b + ": one point, a bool");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        zonotope_points(b);
        ellipsoid_points(b);
        capsule_points(b);
    });
    return test::finish("point masks");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
