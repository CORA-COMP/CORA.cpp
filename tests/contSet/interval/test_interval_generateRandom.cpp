// test_interval_generateRandom - interval generateRandom

#include "contSet/interval/interval.h"
#include "global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

namespace {

void generateRandom_follows_cora(const std::string &b) {
    cora::Rng rng(3);
    const Interval I = Interval::generateRandom(6, rng);
    check(I.inf.shape() == std::vector<int64_t>({6, 1}), b + ": shape");
    bool ordered = true;
    const std::vector<double> lo = I.inf.data(), hi = I.sup.data();
    for (int i = 0; i < 6; ++i) ordered &= lo[i] <= hi[i];
    check(ordered, b + ": inf <= sup");
    // Centre in [-2, 2], radius at most 5.
    const std::vector<double> c = I.center().data(), r = I.rad().data();
    bool in_range = true;
    for (int i = 0; i < 6; ++i) in_range &= std::abs(c[i]) <= 2.0 && r[i] >= 0.0 && r[i] <= 5.0;
    check(in_range, b + ": centre and radius ranges");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { generateRandom_follows_cora(b); });
    return test::finish("interval generateRandom");
}
