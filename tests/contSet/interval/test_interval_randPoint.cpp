// test_interval_randPoint - interval randPoint: points uniform in the box

#include "contSet/interval/interval.h"
#include "global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;
using test::close;

namespace {

/// Random points fill the box: all inside, close to every side, centred.
void randPoint_samples_the_box(const std::string &b) {
    cora::Rng rng(4);
    const int n = 3, N = 2000;
    const Interval I = Interval::generateRandom(n, rng);
    const Tensor P = I.randPoint(N, rng);
    check(P.shape() == std::vector<int64_t>({n, N}), b + ": randPoint shape");
    const std::vector<double> points = P.data(), lo = I.inf.data(), hi = I.sup.data();
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

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { randPoint_samples_the_box(b); });
    return test::finish("interval randPoint");
}
