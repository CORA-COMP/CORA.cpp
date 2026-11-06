// test_interval_radius - interval radius: the radius of the smallest ball about the center
//
// CORA: contSet/@interval/radius.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void radius_is_the_norm_of_rad(const std::string &b) {
    const Interval I(column({0.0, 1.0}), column({2.0, 9.0}));
    check(close(I.radius(), std::vector<double>{std::sqrt(1.0 + 16.0)}), b + ": ||rad||_2");
    const Interval M(Tensor::from_data({0, 1, 2, 3}, {2, 2}),
                     Tensor::from_data({2, 1, 2, 7}, {2, 2}));
    check(close(M.radius(), std::vector<double>{std::sqrt(5.0)}), b + ": over all entries");
    const Interval J(column({0.0, 0.0}), column({1.0, 1.0}));
    const Tensor R = Interval::stack({I, J}).radius();
    check(R.shape() == std::vector<int64_t>({2, 1, 1}), b + ": a batch has a batch shape");
    check(close(R.slice(0, 1, 2).reshape({1}), J.radius().reshape({1})), b + ": a batch member");
}

/// Every point of the box is within the radius of the center, and a vertex is at that distance.
void radius_bounds_the_distance(const std::string &b) {
    cora::Rng rng(12);
    for (int trial = 0; trial < 20; ++trial) {
        const Interval I = Interval::generate_random(rng, 4);
        const double r = I.radius().data()[0];
        const Tensor x = I.rand_point(30, rng), c = I.center();
        const std::vector<double> distances = (x - c).square().sum(-2).sqrt().data();
        double farthest = 0;
        for (const double d : distances) farthest = std::max(farthest, d);
        check(farthest <= r * (1 + 1e-12), b + ": the sampled points are within the radius");
        const Tensor &corner = I.sup();
        check(close((corner - c).square().sum().sqrt().data()[0], r), b + ": a corner attains it");
    }
}

/// radius has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({2.0, 9.0}));
    ChecksScope off(false);
    check(close(I.radius(), std::vector<double>{std::sqrt(17.0)}),
          "radius: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        radius_is_the_norm_of_rad(b);
        radius_bounds_the_distance(b);
    });
    checks_and_their_switch();
    return test::finish("interval radius");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
