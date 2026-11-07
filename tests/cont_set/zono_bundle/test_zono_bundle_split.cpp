// test_zono_bundle_split - ZonoBundle::split: the pieces of a bundle cut at the middle of its box
//
// CORA: contSet/@zonoBundle/split.m
// PROPERTY: contSet/@zonoBundle/split.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The box [-1, 2] x [-1, 1] of the two zonotopes is cut at x = 0.5.
void split_known(const std::string &b) {
    const ZonoBundle zono_bundle = test::two_zonotopes();
    const auto [low, high] = zono_bundle.split(0);
    check(low.parallel_sets() == 3 && high.parallel_sets() == 3, b + ": a member more");
    check(close(low.interval().inf(), {-1, -1}, 1e-12) &&
              close(low.interval().sup(), {0.5, 1}, 1e-12),
          b + ": the lower piece");
    check(close(high.interval().inf(), {0.5, -1}, 1e-12) &&
              close(high.interval().sup(), {2, 1}, 1e-12),
          b + ": the upper piece");
    const auto all = zono_bundle.split();
    check(all.size() == 2 && close(all[1].first.interval().sup(), {2, 0}, 1e-12),
          b + ": a pair per coordinate");
}

/// Stated property: every point of the bundle is in one of the pieces.
void split_property(const std::string &b) {
    Rng rng(66);
    Tensor common;
    const ZonoBundle zono_bundle = test::random_zono_bundle(rng, 3, 2, 4, &common);
    const auto [low, high] = zono_bundle.split(1);
    const Tensor points = test::bundle_points(zono_bundle, rng, 12, common);
    bool covered = true;
    for (int64_t k = 0; k < points.shape().back(); ++k) {
        const Tensor p = points.slice(-1, k, k + 1);
        covered = covered && (low.contains(p) || high.contains(p));
    }
    check(covered, b + ": the pieces cover the bundle");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        split_known(b);
        split_property(b);
    });
    const ZonoBundle zono_bundle = test::two_zonotopes();
    check(test::raises([&] { zono_bundle.split(2); }, Error::wrong_value),
          "split: a coordinate too big");
    check(test::raises([] { ZonoBundle::empty(2).split(); }, Error::empty_set),
          "split: the empty set");
    ChecksScope off(false);
    check(zono_bundle.split(1).first.parallel_sets() == 3, "split: the same with the checks off");
    return test::finish("zono_bundle split");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
