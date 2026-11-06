// test_capsule_interval - Capsule interval: the smallest box around a capsule
//
// CORA: contSet/@capsule/interval.m
// PROPERTY: contSet/@capsule/interval.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
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

/// Hand-computed: c +- (|g| + r) per coordinate; a ball and a point; the empty capsule.
void interval_known(const std::string &b) {
    const Capsule C(column({1.0, 2.0}), column({1.0, -2.0}), 0.5);
    const Interval I = C.interval();
    check(close(I.inf(), {-0.5, -0.5}, 0) && close(I.sup(), {2.5, 4.5}, 0),
          b + ": the segment's reach plus the radius");
    const Interval ball = Capsule(column({1.0, 2.0}), column({0.0, 0.0}), 0.25).interval();
    check(close(ball.inf(), {0.75, 1.75}, 0) && close(ball.sup(), {1.25, 2.25}, 0), b + ": a ball");
    const Interval point = Capsule(column({1.0, 2.0})).interval();
    check(close(point.inf(), {1, 2}, 0) && close(point.sup(), {1, 2}, 0), b + ": a point");
    check(Capsule::empty(2).interval().is_empty(), b + ": the empty capsule");
    const Tensor c = Tensor::from_data({0, 0, 1, 1}, {2, 2, 1});
    const Interval batch = Capsule(c, Tensor::ones({2, 2, 1}), 0.5).interval();
    check(close(batch.inf(), {-1.5, -1.5, -0.5, -0.5}, 0), b + ": a batch");
}

/// Stated property: the box contains random points of the capsule and the corners of the segment's
/// ball reach it (it is the smallest box).
void interval_property(const std::string &b) {
    Rng rng(13);
    for (int k = 0; k < 8; ++k) {
        const Capsule C = test::random_capsule(rng, 3);
        const Interval I = C.interval();
        check(I.contains(C.center()), b + ": the center is in the box");
        const Tensor points = test::sample_points(C, rng, 20);
        for (int j = 0; j < 20; ++j)
            check(I.contains(points.slice(-1, j, j + 1)), b + ": a sampled point is in the box");
        // the point of the capsule furthest in direction i touches the box
        const std::vector<double> c = C.c().data(), g = C.generator().data();
        const double r = C.r().data()[0];
        for (int i = 0; i < 3; ++i) {
            const double top = c[i] + std::abs(g[i]) + r;
            check(close(I.sup().data()[i], top, 1e-12), b + ": the bound is attained");
        }
    }
}

/// A batch with an empty member has no box; with the checks off a valid call gives the same box.
void checks_and_their_switch() {
    const Tensor c = Tensor::zeros({2, 2, 1});
    // the checked constructor refuses the radius -1, so the batch is built with the checks off
    const Capsule mixed = [&] {
        ChecksScope off(false);
        return Capsule(c, c, Tensor::from_data({-1.0, 1.0}, {2, 1, 1}));
    }();
    check(test::raises([&] { mixed.interval(); }, Error::not_supported),
          "interval: a batch with an empty member has no box");
    const Capsule C(column({1.0, 2.0}), column({1.0, -2.0}), 0.5);
    const Interval on = C.interval();
    ChecksScope off(false);
    check(close(C.interval().sup(), on.sup(), 0), "interval: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        interval_known(b);
        interval_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
