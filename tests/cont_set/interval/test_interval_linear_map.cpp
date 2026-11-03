// test_interval_linear_map - interval linear_map: the interval hull of the image
//
// CORA: contSet/@interval/mtimes.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
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
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    std::vector<double> x(lo.size());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = lo[i] + t[i] * (hi[i] - lo[i]);
    return x;
}

/// `M * I` is the interval hull of the image: it holds the images of the box's points, and
/// along each axis it reaches as far as the image does.
void mtimes_is_the_hull_of_the_image(const std::string &b) {
    cora::Rng rng(2);
    const int n = 3;
    const Interval I = Interval::generate_random(rng, n);
    std::vector<double> M(n * n);
    rng.normal(M.data(), M.size(), 1.0);
    const Interval image = I.linear_map(Tensor::from_data(M, {n, n}));

    for (int trial = 0; trial < 50; ++trial) {
        const std::vector<double> x = point_of(I, random_vector(rng, n, 0.0, 1.0));
        std::vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) y[i] += M[i * n + j] * x[j];
        check(image.contains(column(y)), b + ": an image point left M * I");
    }
    // For a matrix that is a permutation with signs the hull is the image itself.
    const Interval flipped =
        I.linear_map(Tensor({{0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}}));
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    check(close(flipped.inf(), std::vector<double>{-hi[1], lo[0], lo[2]}),
          b + ": exact for a rotation");
    check(close(flipped.sup(), std::vector<double>{-lo[1], hi[0], hi[2]}),
          b + ": exact for a rotation");
}

/// A matrix of the wrong width is refused; with the checks off the map is the same.
void checks_and_their_switch() {
    const Interval S = Interval(test::column({0, -1}), test::column({1, 1}));
    const Tensor M({{1.0, 2.0}, {0.0, 1.0}}), wide({{1.0, 2.0, 3.0}});
    check(test::raises([&] { S.linear_map(wide); }, Error::dimension_mismatch),
          "linear_map: a matrix of three columns");
    const Tensor on = S.linear_map(M).center();
    ChecksScope off(false);
    check(test::close(S.linear_map(M).center(), on, 0), "linear_map: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { mtimes_is_the_hull_of_the_image(b); });
    checks_and_their_switch();
    return test::finish("interval linear_map");
}
