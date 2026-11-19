// test_ellipsoid_ball_radius - Ellipsoid ball_radius: the largest semi-axis
//
// CORA.cpp only: CORA's ellipsoid.radius is the semi-axes (test_ellipsoid_semi_axes).
// Authors:       Claude Opus 5.5
// Written:       19-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/global/rng.h"
#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the largest semi-axis, one number per member; 0 for the empty ellipsoid.
void ball_radius_known(const std::string &b) {
    const Ellipsoid E(vec({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 4.0}}));
    check(close(E.ball_radius(), {2.0}, 1e-14) && E.ball_radius().ndim() == 0,
          b + ": the largest semi-axis, a number");
    const Tensor Qs = Tensor::stack({Tensor({{9.0, 0.0}, {0.0, 1.0}}), Tensor::zeros({2, 2})}, 0);
    check(close(Ellipsoid(Qs).ball_radius(), {3.0, 0.0}, 1e-14) &&
              Ellipsoid(Qs).ball_radius().shape() == Shape{2},
          b + ": a batch, a point has 0");
    check(close(Ellipsoid::empty(2).ball_radius(), {0.0}, 0), b + ": empty");
}

/// Stated property: the boundary points are within the radius of the center.
void ball_radius_property(const std::string &b) {
    Rng rng(106);
    for (int k = 0; k < 4; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3, k % 2 ? 2 : -1);
        const Tensor P = test::ellipsoid_points(E, rng, 30, true) - E.q();
        check(P.square().sum(-1).sqrt().max().item() <= E.ball_radius().item() + 1e-9,
              b + ": inside the ball");
    }
}

/// No input to get wrong; with the checks off the call is the same.
void switch_off() {
    const Ellipsoid E(Tensor({{2.0, 1.0}, {1.0, 2.0}}));
    const Tensor on = E.ball_radius();
    ChecksScope off(false);
    check(close(E.ball_radius(), on, 0), "ball_radius: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        ball_radius_known(b);
        ball_radius_property(b);
    });
    switch_off();
    return test::finish("ellipsoid ball_radius");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
