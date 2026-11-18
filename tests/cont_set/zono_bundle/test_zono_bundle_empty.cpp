// test_zono_bundle_empty - ZonoBundle::empty: the bundle without points
//
// CORA: contSet/@zonoBundle/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The empty bundle is the points 0 and e_1; no point lies in both.
void empty_set_data(const std::string &b) {
    const ZonoBundle E = ZonoBundle::empty(3);
    check(E.dim() == 3 && E.parallel_sets() == 2, b + ": dimension 3, two members");
    // layout: #163 step 2
    check(test::close(cora::detail::cora_center(E.zonotopes()[0]), {0.0, 0.0, 0.0}, 0),
          b + ": the origin");
    // layout: #163 step 2
    check(test::close(cora::detail::cora_center(E.zonotopes()[1]), {1.0, 0.0, 0.0}, 0),
          b + ": e_1");
    // layout: #163 step 2
    check(cora::detail::cora_generators(E.zonotopes()[0]).shape() == Shape{3, 0},
          b + ": no generators");
    check(E.interval().is_empty(), b + ": its box is empty");
    check(E.center().shape() == Shape{3, 0}, b + ": its center has no column");
    check(!E.contains(column({0.0, 0.0, 0.0})) && !E.contains(column({1.0, 0.0, 0.0})),
          b + ": it contains no point");
    check(!E.is_full_dim(), b + ": it has no interior");
}

/// A dimension below one is refused; the same call with the checks off.
void checks_and_their_switch() {
    check(test::raises([] { ZonoBundle::empty(0); }, Error::wrong_value), "empty: dimension 0");
    check(test::raises([] { ZonoBundle::empty(-1); }, Error::wrong_value), "empty: negative");
    ChecksScope off(false);
    check(ZonoBundle::empty(2).dim() == 2, "empty: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(empty_set_data);
    checks_and_their_switch();
    return test::finish("zono_bundle empty");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
