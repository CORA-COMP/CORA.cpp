// test_zono_bundle_is_full_dim - ZonoBundle::is_full_dim: whether the bundle has an interior
//
// CORA: contSet/@zonoBundle/isFullDim.m
// PROPERTY: contSet/@zonoBundle/isFullDim.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The unit square with center (c1, c2).
Zonotope square(double c1, double c2) {
    return Zonotope(column({c1, c2}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
}

/// Overlapping squares have an interior; squares that touch share a segment, disjoint ones
/// nothing; a flat member is not full-dimensional.
void known_bundles(const std::string &b) {
    check(ZonoBundle({square(0.0, 0.0), square(0.25, 0.25)}).is_full_dim(), b + ": overlapping");
    check(!ZonoBundle({square(0.0, 0.0), square(1.0, 0.0)}).is_full_dim(), b + ": touching");
    check(!ZonoBundle({square(0.0, 0.0), square(3.0, 0.0)}).is_full_dim(), b + ": disjoint");
    const Zonotope flat(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    check(!ZonoBundle({square(0.0, 0.0), flat}).is_full_dim(), b + ": a flat member");
    check(ZonoBundle({square(0.0, 0.0)}).is_full_dim(), b + ": one full-dimensional member");
    check(Set(ZonoBundle({square(0.0, 0.0), square(0.25, 0.0)}))->is_full_dim(),
          b + ": through the set interface");
}

/// Stated property: a bundle that contains a ball around its center is full-dimensional.
void ball_around_the_center(const std::string &b) {
    Rng rng(25);
    for (int trial = 0; trial < 4; ++trial) {
        const ZonoBundle bundle = test::random_zono_bundle(rng, 2, 2, 4);
        const Tensor c = bundle.center();
        const Tensor r = c.full_like({2, 1}, 1e-4);
        const bool ball = bundle.contains(Interval(c - r, c + r));
        check(!ball || bundle.is_full_dim(), b + ": a ball inside means full-dimensional");
    }
}

/// A negative tolerance and a batch are refused; the same call with the checks off.
void checks_and_their_switch() {
    const ZonoBundle bundle({square(0.0, 0.0), square(0.25, 0.0)});
    check(test::raises([&] { bundle.is_full_dim(-1.0); }, Error::wrong_value),
          "is_full_dim: tol < 0");
    check(test::raises(
              [&] { ZonoBundle({Zonotope::stack({square(0, 0), square(0, 0)})}).is_full_dim(); },
              Error::not_supported),
          "is_full_dim: a batch");
    // the set interface refuses a batch with the checks off too
    ChecksScope off(false);
    const ZonoBundle batch({Zonotope::stack({square(0, 0), square(0, 0)})});
    check(test::raises([&] { Set(batch)->is_full_dim(); }, Error::not_supported),
          "is_full_dim: a batch through the set interface, checks off");
    check(bundle.is_full_dim(), "is_full_dim: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        known_bundles(b);
        ball_around_the_center(b);
    });
    checks_and_their_switch();
    return test::finish("zono_bundle is_full_dim");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
