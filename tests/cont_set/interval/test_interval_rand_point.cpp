// test_interval_rand_point - interval rand_point: points uniform in the box
//
// CORA: contSet/@interval/randPoint_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;
using test::close;

namespace {

/// Random points fill the box: all inside, close to every side, centred.
void rand_point_samples_the_box(const std::string &b) {
    cora::Rng rng(4);
    const int n = 3, N = 2000;
    const Interval I = Interval::generate_random(n, rng);
    const Tensor P = I.rand_point(N, rng);
    check(P.shape() == std::vector<int64_t>({n, N}), b + ": rand_point shape");
    const std::vector<double> points = P.data(), lo = I.inf().data(), hi = I.sup().data();
    bool inside = true;
    for (int i = 0; i < n; ++i) {
        double smallest = 1e300, largest = -1e300, mean = 0.0;
        for (int j = 0; j < N; ++j) {
            const double x = points[i * N + j];
            inside &= x >= lo[i] && x <= hi[i];
            smallest = std::min(smallest, x);
            largest = std::max(largest, x);
            mean += x / N;
        }
        const double width = hi[i] - lo[i];
        check(smallest - lo[i] < 0.02 * width + 1e-12 && hi[i] - largest < 0.02 * width + 1e-12,
              b + ": the points miss a side of the box");
        check(std::abs(mean - (lo[i] + hi[i]) / 2) < 0.05 * width + 1e-12,
              b + ": the points are not centred");
    }
    check(inside, b + ": a random point left the box");
}

/// A negative number of points is refused; with the checks off the draw is the same.
void checks_and_their_switch() {
    const Interval S = Interval(test::column({0, -1}), test::column({1, 1}));
    cora::Rng rng(3), again(3);
    check(test::raises([&] { S.rand_point(-1, rng); }, Error::wrong_value), "rand_point: N = -1");
    const Tensor on = S.rand_point(4, rng);
    ChecksScope off(false);
    check(test::close(S.rand_point(4, again), on, 0), "rand_point: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { rand_point_samples_the_box(b); });
    checks_and_their_switch();
    return test::finish("interval rand_point");
}
