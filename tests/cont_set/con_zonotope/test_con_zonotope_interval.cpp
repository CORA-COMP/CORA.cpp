// test_con_zonotope_interval - ConZonotope::interval: the smallest box around the set
//
// CORA: contSet/@conZonotope/interval.m
// PROPERTY: contSet/@conZonotope/interval.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
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

/// Hand-computed: a zonotope's box; the exact range of a segment; a set without points.
void interval_known(const std::string &b) {
    const ConZonotope plain(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 3.0}}));
    check(close(plain.interval().inf(), {0, -1}, 0) && close(plain.interval().sup(), {2, 5}, 0),
          b + ": without constraints the box of the zonotope");
    // the segment from (0, 1) to (1, 0): x1 in [0, 1] although xi1 reaches [-1, 1]
    const ConZonotope segment(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), column({1.0}));
    const Interval I = segment.interval();
    check(close(I.inf(), {0, 0}, 1e-9) && close(I.sup(), {1, 1}, 1e-9),
          b + ": the exact range of a segment");
    const ConZonotope none(column({0.0}), Tensor({{1.0}}), Tensor({{1.0}}), column({3.0}));
    check(none.interval().is_empty(), b + ": no point: the empty box");
    check(Set(segment)->interval().sup().shape() == Shape{2, 1}, b + ": through the set interface");
    // zero generators and zero rows do not change the box
    const ConZonotope padded(column({0.0, 0.0}), Tensor({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}),
                             Tensor({{1.0, 1.0, 0.0}, {0.0, 0.0, 0.0}}), column({1.0, 0.0}));
    check(close(padded.interval().sup(), {1, 1}, 1e-9),
          b + ": zero generators and rows are dropped");
}

/// Stated property: the box holds every point of the set, and is no larger than the box that
/// forgets the constraints.
void interval_property(const std::string &b) {
    cora::Rng rng(22);
    for (int k = 0; k < 6; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, 2);
        const Interval I = cz.interval();
        const Tensor points = test::sample_points(cz, rng, 200);
        check(test::inside_box(points, I), b + ": every sampled point is in the box");
        // never larger than the box of the zonotope that forgets the constraints
        const Interval loose = ConZonotope(cz.c(), cz.generators()).interval();
        check(test::inside_box(I.inf(), loose) && test::inside_box(I.sup(), loose),
              b + ": inside the box of the zonotope");
    }
}

/// A batch with constraints is refused; with the checks off a valid call gives the same box.
void checks_and_their_switch() {
    const ConZonotope batch(Tensor::zeros({2, 1, 1}), Tensor::ones({2, 1, 2}),
                            Tensor::ones({2, 1, 2}), Tensor::zeros({2, 1, 1}));
    check(test::raises([&] { batch.interval(); }, Error::not_supported),
          "interval: a batch with constraints");
    const ConZonotope plain(Tensor::zeros({2, 1, 1}), Tensor::ones({2, 1, 2}));
    check(plain.interval().sup().shape() == Shape{2, 1, 1},
          "interval: a batch without constraints");
    const ConZonotope segment(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), column({1.0}));
    const Tensor on = segment.interval().sup();
    ChecksScope off(false);
    check(test::close(segment.interval().sup(), on, 0), "interval: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        interval_known(b);
        interval_property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
