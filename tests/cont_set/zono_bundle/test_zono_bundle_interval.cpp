// test_zono_bundle_interval - ZonoBundle::interval: the box around a bundle
//
// CORA: contSet/@zonoBundle/interval.m
//
// CORA authors: Matthias Althoff
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

/// The boxes [-2, 2] x [-1, 1] and [-1, 3] x [-1, 1] meet in [-1, 2] x [-1, 1].
void intersection_of_boxes(const std::string &b) {
    const Zonotope Z1(column({0.0, 0.0}), Tensor({{2.0, 0.0}, {0.0, 1.0}}));
    const Zonotope Z2(column({1.0, 0.0}), Tensor({{1.0, 1.0}, {0.0, 1.0}}));
    const Interval I = ZonoBundle({Z1, Z2}).interval();
    check(close(I.inf(), {-1.0, -1.0}, 0) && close(I.sup(), {2.0, 1.0}, 0), b + ": [-1, 2] x [-1, 1]");
    check(ZonoBundle({Z1}).interval().inf().data() == Z1.interval().inf().data(),
          b + ": one member: its own box");
    const Zonotope far(column({10.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(ZonoBundle({Z1, far}).interval().is_empty(), b + ": disjoint boxes give the empty box");
    check(Set(ZonoBundle({Z1, Z2}))->interval().is_empty() == false, b + ": through the interface");
}

/// A batch of bundles gives the batch of boxes, member by member.
void batch(const std::string &b) {
    Rng rng(24);
    const Zonotope A1 = Zonotope::generate_random(rng, 2, std::nullopt, 3),
                   A2 = Zonotope::generate_random(rng, 2, std::nullopt, 3);
    const Zonotope B1 = Zonotope::generate_random(rng, 2, std::nullopt, 3),
                   B2 = Zonotope::generate_random(rng, 2, std::nullopt, 3);
    const Interval boxes = ZonoBundle({Zonotope::stack({A1, A2}), Zonotope::stack({B1, B2})}).interval();
    const Interval second = ZonoBundle({A2, B2}).interval();
    if (second.is_empty() || boxes.is_empty()) return;
    check(close(boxes.inf().slice(0, 1, 2).squeeze(0), second.inf(), 0), b + ": member 2 of a batch");
}

/// The call takes no input; with the checks off the box is the same.
void switch_off() {
    const Zonotope Z(column({0.0}), Tensor({{1.0}}));
    ChecksScope off(false);
    check(close(ZonoBundle({Z, Z}).interval().sup(), {1.0}, 0), "interval: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        intersection_of_boxes(b);
        batch(b);
    });
    switch_off();
    return test::finish("zono_bundle interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
