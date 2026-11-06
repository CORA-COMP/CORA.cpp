// test_zono_bundle_is_bounded - ZonoBundle::is_bounded: always true
//
// CORA: contSet/@zonoBundle/isBounded.m
// CORA authors: Mark Wetzlinger
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

/// Every bundle is bounded: a single set, the empty one and a batch.
void always_bounded(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 3.0}}));
    check(ZonoBundle({Z, Z}).is_bounded(), b + ": a bundle is bounded");
    check(ZonoBundle::empty(2).is_bounded(), b + ": the empty bundle is bounded");
    check(Set(ZonoBundle({Z}))->is_bounded(), b + ": through the set interface");
    check(ZonoBundle({Zonotope::stack({Z, Z})}).is_bounded(), b + ": a batch is bounded");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(always_bounded);
    // the question takes no input to get wrong; with the checks off the answer is the same
    ChecksScope off(false);
    check(ZonoBundle::empty(1).is_bounded(), "is_bounded: the same with the checks off");
    return test::finish("zono_bundle is_bounded");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
