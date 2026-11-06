// test_capsule_radius - Capsule radius: |g| + r
//
// CORA: contSet/@capsule/radius.m
// PROPERTY: contSet/@capsule/radius.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: |g| + r, for a batch, and the lengths against a brute-force bound.
void radius_known(const std::string &b) {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(close(C.radius(), {5.5}, 1e-14), b + ": |(3, 4)| + 0.5");
    check(close(Capsule(column({1.0})).radius(), {0.0}, 0), b + ": a point");
    const Tensor c = Tensor::zeros({2, 2, 1});
    const Capsule batch(c, Tensor::from_data({3, 4, 0, 1}, {2, 2, 1}),
                        Tensor::from_data({0.5, 0.25}, {2, 1, 1}));
    check(close(batch.radius(), {5.5, 1.25}, 1e-14), b + ": a batch");
}

/// Stated property: every point of the capsule is within the radius of the center.
void radius_property(const std::string &b) {
    Rng rng(41);
    for (int k = 0; k < 5; ++k) {
        const Capsule C = test::random_capsule(rng, 3);
        const Tensor P = test::sample_points(C, rng, 40) - C.c();
        const double far = P.square().sum(0).sqrt().max().item();
        check(far <= C.radius().item() + 1e-12, b + ": the points lie in the ball of radius()");
    }
}

/// The checks of the entry have nothing to refuse: the call works with them off too.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    const Tensor on = C.radius();
    ChecksScope off(false);
    check(close(C.radius(), on, 0), "radius: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        radius_known(b);
        radius_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule radius");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
