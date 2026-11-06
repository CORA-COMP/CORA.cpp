// test_con_zonotope_split - ConZonotope::split: two sets that cover the set
//
// CORA: contSet/@conZonotope/split.m
// PROPERTY: contSet/@conZonotope/split.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the segment from (0, 1) to (1, 0) cut at x1 = 0.5 gives two halves.
void split_known(const std::string &b) {
    const ConZonotope seg(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                          Tensor({{1.0, 1.0}}), column({1.0}));
    const auto [low, high] = seg.split(0);
    check(close(low.interval().sup(), {0.5, 1.0}, 1e-9) &&
              close(low.interval().inf(), {0, 0.5}, 1e-9),
          b + ": the lower half");
    check(close(high.interval().inf(), {0.5, 0}, 1e-9) &&
              close(high.interval().sup(), {1, 0.5}, 1e-9),
          b + ": the upper half");
    check(seg.split().size() == 2, b + ": one pair per coordinate");
}

/// Stated property: every sampled point is in one of the halves, and each half is in its half of
/// the box.
void split_property(const std::string &b) {
    cora::Rng rng(47);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, 2);
        const Interval box = cz.interval();
        const auto [low, high] = cz.split(1);
        const double middle = box.center().data()[1];
        check(low.interval().sup().data()[1] <= middle + 1e-8 &&
                  high.interval().inf().data()[1] >= middle - 1e-8,
              b + ": each half lies on its side");
        const Tensor points = test::sample_points(cz, rng, 12);
        for (int64_t j = 0; j < points.shape()[1]; ++j) {
            const Tensor p = points.slice(1, j, j + 1);
            check(low.contains(p) || high.contains(p), b + ": a point is in a half");
        }
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        split_known(b);
        split_property(b);
    });
    const ConZonotope a(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { a.split(1); }, Error::wrong_value), "split: a coordinate");
    check(test::raises([&] { ConZonotope::empty(2).split(0); }, Error::empty_set), "split: empty");
    ChecksScope off(false);
    check(a.split(0).first.dim() == 1, "split: the same with the checks off");
    return test::finish("con_zonotope split");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
