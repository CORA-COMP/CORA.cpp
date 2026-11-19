// test_ellipsoid_center - Ellipsoid::center: the stored center
//
// CORA: contSet/@ellipsoid/center.m
// PROPERTY: contSet/@ellipsoid/center.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The center is q, in the ellipsoid, for a batch and for a degenerate set.
void center_known(const std::string &b) {
    const Ellipsoid E(vec({1.0, 2.0}), Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    check(close(E.center(), {1, 2}, 0), b + ": q");
    const Ellipsoid batch(Tensor::ones({2, 2}), Tensor::zeros({2, 2, 2}));
    check(close(batch.center(), Tensor::ones({2, 2}), 0), b + ": a batch");
    const Tensor none = Ellipsoid::empty(2).center();
    check(none.shape() == Shape{2} && std::isnan(none.data()[0]), b + ": the empty ellipsoid");
}

/// Stated property: the center is in the ellipsoid, at norm 0.
void center_property(const std::string &b) {
    cora::Rng rng(11);
    for (int k = 0; k < 6; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3, k % 2 == 0 ? -1 : 2);
        check(E.contains(E.center(), "exact"), b + ": the center is in the set");
        check(test::finite_near(E.ellipsoid_norm(E.center() - E.q()), 0.0, 1e-12),
              b + ": at norm zero");
    }
}

/// The center has no input to get wrong; with the checks off it is the same.
void switch_off() {
    const Ellipsoid E(vec({3.0}), Tensor({{1.0}}));
    const Tensor on = E.center();
    ChecksScope off(false);
    check(test::close(E.center(), on, 0), "center: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        center_known(b);
        center_property(b);
    });
    switch_off();
    return test::finish("ellipsoid center");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
