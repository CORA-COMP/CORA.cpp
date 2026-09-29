// The interval on every backend: each operation against its definition.

#include "contSet/interval/interval.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;

namespace {

Tensor column(const std::vector<double> &v) { return Tensor::from_data(v, {int64_t(v.size()), 1}); }

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

void basics(const std::string &b) {
    const Interval I(column({-1.0, 0.0, 2.0}), column({3.0, 0.5, 2.0}));
    check(I.dim() == 3, b + ": dim");
    check(close(I.center(), std::vector<double>{1.0, 0.25, 2.0}), b + ": center");
    check(close(I.rad(), std::vector<double>{2.0, 0.25, 0.0}), b + ": radius");
    check(close(I.interval().inf, I.inf) && close(I.interval().sup, I.sup), b + ": its own hull");
}

/// The support of a box along `d` is the best corner, `Σ max(d_i lo_i, d_i hi_i)`.
void support_func_matches_the_corners(const std::string &b) {
    cora::Rng rng(1);
    for (const int n : {1, 2, 4}) {
        const Interval I = Interval::generate_random(n, rng);
        const std::vector<double> lo = I.inf.data(), hi = I.sup.data();
        for (int k = 0; k < 5; ++k) {
            std::vector<double> d(n);
            rng.normal(d.data(), d.size(), 1.0);
            double want = 0.0;
            for (int i = 0; i < n; ++i) want += std::max(d[i] * lo[i], d[i] * hi[i]);
            check(close(I.support_func(column(d)).data()[0], want, 1e-12),
                  b + ": supportFunc in " + std::to_string(n) + "d");
        }
    }
}

/// `M * I` is the interval hull of the image: it holds the images of the box's points, and
/// along each axis it reaches as far as the image does.
void mtimes_is_the_hull_of_the_image(const std::string &b) {
    cora::Rng rng(2);
    const int n = 3;
    const Interval I = Interval::generate_random(n, rng);
    std::vector<double> M(n * n);
    rng.normal(M.data(), M.size(), 1.0);
    const Interval image = I.mtimes(Tensor::from_data(M, {n, n}));

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
    check(close(flipped.inf, std::vector<double>{-hi[1], lo[0], lo[2]}), b + ": exact for a rotation");
    check(close(flipped.sup, std::vector<double>{-lo[1], hi[0], hi[2]}), b + ": exact for a rotation");
}

void plus_is_the_minkowski_sum(const std::string &b) {
    const Interval A(column({0.0, 1.0}), column({1.0, 3.0})), B(column({-2.0, 0.5}), column({0.0, 0.5}));
    const Interval S = A.plus(B);
    check(close(S.inf, std::vector<double>{-2.0, 1.5}) && close(S.sup, std::vector<double>{1.0, 3.5}),
          b + ": bounds add");
}

void contains_a_point(const std::string &b) {
    const Interval I(column({0.0, -1.0}), column({2.0, 1.0}));
    check(I.contains(column({1.0, 0.0})), b + ": an inner point");
    check(I.contains(column({0.0, 1.0})), b + ": a corner belongs to the box");
    check(!I.contains(column({2.1, 0.0})), b + ": past the upper bound");
    check(!I.contains(column({1.0, -1.5})), b + ": below the lower bound");
}

/// As in CORA an interval can hold matrices; centre and radius are then elementwise.
void bounds_can_be_matrices(const std::string &b) {
    const Interval M(Tensor({{-1.0, 0.0}, {2.0, -3.0}}), Tensor({{1.0, 4.0}, {2.0, 0.0}}));
    check(close(M.center(), std::vector<double>{0.0, 2.0, 2.0, -1.5}), b + ": matrix centre");
    check(close(M.rad(), std::vector<double>{1.0, 2.0, 0.0, 1.5}), b + ": matrix radius");
    check(M.center().shape() == std::vector<int64_t>({2, 2}), b + ": matrix shape");
}

void generate_random_follows_cora(const std::string &b) {
    cora::Rng rng(3);
    const Interval I = Interval::generate_random(6, rng);
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

/// Random points fill the box: all inside, close to every side, centred.
void rand_point_samples_the_box(const std::string &b) {
    cora::Rng rng(4);
    const int n = 3, N = 2000;
    const Interval I = Interval::generate_random(n, rng);
    const Tensor P = I.rand_point(N, rng);
    check(P.shape() == std::vector<int64_t>({n, N}), b + ": rand_point shape");
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
        check(std::abs(mean - (lo[i] + hi[i]) / 2) < 0.05 * width + 1e-12, b + ": the points are not centred");
    }
    check(inside, b + ": a random point left the box");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        basics(b);
        support_func_matches_the_corners(b);
        mtimes_is_the_hull_of_the_image(b);
        plus_is_the_minkowski_sum(b);
        contains_a_point(b);
        bounds_can_be_matrices(b);
        rand_point_samples_the_box(b);
        generate_random_follows_cora(b);
    });
    return test::finish("interval");
}
