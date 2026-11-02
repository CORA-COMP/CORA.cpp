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
    const Interval I = Interval::generate_random(6, rng);
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

/// A dimension below one is refused; with the checks off the draw is the same.
void checks_and_their_switch() {
    cora::Rng rng(4), again(4);
    check(test::raises([&] { Interval::generate_random(0, rng); }, Error::wrong_value),
          "generate_random: dimension 0");
    const Tensor on = Interval::generate_random(3, rng).center();
    ChecksScope off(false);
    check(test::close(Interval::generate_random(3, again).center(), on, 0),
          "generate_random: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { generate_random_follows_cora(b); });
    checks_and_their_switch();
    return test::finish("interval generate_random");
}
