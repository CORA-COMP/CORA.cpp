// test_interval_supportFunc - interval supportFunc: the best corner of the box

#include "contSet/interval/interval.h"
#include "global/rng.h"
#include "testing.h"

#include <algorithm>

using namespace cora::ct;
using test::check;
using test::close;
using test::column;

namespace {

/// The support of a box along `d` is the best corner, `Σ max(d_i lo_i, d_i hi_i)`.
void supportFunc_matches_the_corners(const std::string &b) {
    cora::Rng rng(1);
    for (const int n : {1, 2, 4}) {
        const Interval I = Interval::generateRandom(n, rng);
        const std::vector<double> lo = I.inf.data(), hi = I.sup.data();
        for (int k = 0; k < 5; ++k) {
            std::vector<double> d(n);
            rng.normal(d.data(), d.size(), 1.0);
            double want = 0.0;
            for (int i = 0; i < n; ++i) want += std::max(d[i] * lo[i], d[i] * hi[i]);
            check(close(I.supportFunc(column(d)).data()[0], want, 1e-12),
                  b + ": supportFunc in " + std::to_string(n) + "d");
        }
    }
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { supportFunc_matches_the_corners(b); });
    return test::finish("interval supportFunc");
}
