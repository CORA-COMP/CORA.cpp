// test_interval_generate_random - interval generate_random
//
// CORA: contSet/@interval/generateRandom.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

namespace {

void generate_random_follows_cora(const std::string &b) {
    cora::Rng rng(3);
    const Interval I = Interval::generate_random(rng, 6);
    check(I.inf().shape() == std::vector<int64_t>({6, 1}), b + ": shape");
    bool ordered = true;
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    for (int i = 0; i < 6; ++i) ordered &= lo[i] <= hi[i];
    check(ordered, b + ": inf <= sup");
    // Centre in [-2, 2], radius at most 5.
    const std::vector<double> c = I.center().data(), r = I.rad().data();
    bool in_range = true;
    for (int i = 0; i < 6; ++i) in_range &= std::abs(c[i]) <= 2.0 && r[i] >= 0.0 && r[i] <= 5.0;
    check(in_range, b + ": centre and radius ranges");
}

/// The options: the center and the radii given, the dimension taken from them.
void generate_random_options(const std::string &b) {
    cora::Rng rng(5);
    const Tensor center = test::column({1.0, -1.0, 0.5});
    const Interval I = Interval::generate_random(rng, std::nullopt, center, test::column({2, 0, 4}));
    check(I.dim() == 3, b + ": the dimension of the center");
    const std::vector<double> c = I.center().data(), r = I.rad().data();
    check(close(I.center(), {1.0, -1.0, 0.5}, 1e-12), b + ": the given center");
    check(r[0] <= 1.0 && r[1] == 0.0 && r[2] <= 2.0, b + ": radius at most max_radius / 2");
    const Interval S = Interval::generate_random(rng, 4, std::nullopt, Tensor::from_data({6.0}, {1, 1}));
    bool small = true;
    for (const double x : S.rad().data()) small &= x >= 0.0 && x <= 3.0;
    check(S.dim() == 4 && small, b + ": one number for all radii");
    check(test::raises([&] { Interval::generate_random(rng, 2, center); }, Error::wrong_value),
          b + ": the center does not match the dimension");
    check(test::raises([&] { Interval::generate_random(rng, std::nullopt, std::nullopt,
                                                       test::column({-1.0})); },
                       Error::wrong_value),
          b + ": a negative radius");
}

/// A dimension below one is refused; with the checks off the draw is the same.
void checks_and_their_switch() {
    cora::Rng rng(4), again(4);
    check(test::raises([&] { Interval::generate_random(rng, 0); }, Error::wrong_value),
          "generate_random: dimension 0");
    const Tensor on = Interval::generate_random(rng, 3).center();
    ChecksScope off(false);
    check(test::close(Interval::generate_random(again, 3).center(), on, 0),
          "generate_random: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        generate_random_follows_cora(b);
        generate_random_options(b);
    });
    checks_and_their_switch();
    return test::finish("interval generate_random");
}
