// test_interval_mtimes - interval mtimes: the interval hull of the image

#include "contSet/interval/interval.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;
using test::column;

namespace {

std::vector<double> random_vector(cora::Rng &rng, int n, double lo, double hi) {
    std::vector<double> v(n);
    rng.uniform(v.data(), v.size(), lo, hi);
    return v;
}

/// A point of the box, from a fraction of each side.
std::vector<double> point_of(const Interval &I, const std::vector<double> &t) {
    const std::vector<double> lo = I.inf.data(), hi = I.sup.data();
    std::vector<double> x(lo.size());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = lo[i] + t[i] * (hi[i] - lo[i]);
    return x;
}

/// `M * I` is the interval hull of the image: it holds the images of the box's points, and
/// along each axis it reaches as far as the image does.
void mtimes_is_the_hull_of_the_image(const std::string &b) {
    cora::Rng rng(2);
    const int n = 3;
    const Interval I = Interval::generateRandom(n, rng);
    std::vector<double> M(n * n);
    rng.normal(M.data(), M.size(), 1.0);
    const Interval image = I.mtimes(Tensor::fromData(M, {n, n}));

    for (int trial = 0; trial < 50; ++trial) {
        const std::vector<double> x = point_of(I, random_vector(rng, n, 0.0, 1.0));
        std::vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) y[i] += M[i * n + j] * x[j];
        check(image.contains(column(y)), b + ": an image point left M * I");
    }
    // For a matrix that is a permutation with signs the hull is the image itself.
    const Interval flipped = I.mtimes(Tensor({{0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}}));
    const std::vector<double> lo = I.inf.data(), hi = I.sup.data();
    check(close(flipped.inf, std::vector<double>{-hi[1], lo[0], lo[2]}),
          b + ": exact for a rotation");
    check(close(flipped.sup, std::vector<double>{-lo[1], hi[0], hi[2]}),
          b + ": exact for a rotation");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { mtimes_is_the_hull_of_the_image(b); });
    return test::finish("interval mtimes");
}
