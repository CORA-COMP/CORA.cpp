// test_zono_bundle_center - ZonoBundle::center: a point of the bundle near its middle
//
// CORA: contSet/@zonoBundle/center.m
// PROPERTY: contSet/@zonoBundle/center.m
//
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A single member has the zonotope's own center.
void single_member(const std::string &b) {
    const Zonotope Z(column({1.0, -2.0}), Tensor({{1.0, 0.5}, {0.0, 1.0}}));
    check(close(ZonoBundle({Z}).center(), Z.center(), 0), b + ": the center of the member");
}

/// Stated property: the center lies in the bundle (CORA's is a Chebyshev center of the factors).
void center_is_in_the_bundle(const std::string &b) {
    Rng rng(22);
    for (int trial = 0; trial < 5; ++trial) {
        const ZonoBundle bundle = test::random_zono_bundle(rng, 2 + trial % 2, 2, 3);
        check(bundle.center().shape() == Shape{bundle.dim(), 1} && bundle.contains(bundle.center()),
              b + ": the center is a point of the bundle");
    }
}

/// An empty bundle has a center without columns; the call takes no input, and runs with the
/// checks off.
void empty_bundle_and_switch() {
    check(ZonoBundle::empty(2).center().shape() == Shape{2, 0}, "center: empty bundle, no column");
    ChecksScope off(false);
    Rng rng(23);
    const ZonoBundle bundle = test::random_zono_bundle(rng, 2, 2, 3);
    check(bundle.contains(bundle.center()), "center: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        single_member(b);
        center_is_in_the_bundle(b);
    });
    empty_bundle_and_switch();
    return test::finish("zono_bundle center");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
