// test_zonotope_rand_point - zonotope rand_point: standard points in the set, extreme points at its
// corners
//
// CORA: contSet/@zonotope/randPoint_.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::random_direction;
using test::support;

namespace {

double dot_column(const std::vector<double> &points, int n, int N, int j,
                  const std::vector<double> &d) {
    double v = 0.0;
    for (int i = 0; i < n; ++i) v += d[i] * points[i * N + j];
    return v;
}

/// Random points lie in the set, spread over it, and centre on it.
void rand_point_samples_the_set(const std::string &b) {
    cora::Rng rng(8);
    const int n = 3, m = 4, N = 3000;
    const Zonotope Z = Zonotope::generate_random(n, m, rng);
    const Tensor P = Z.rand_point(N, rng);
    check(P.shape() == std::vector<int64_t>({n, N}), b + ": rand_point shape");
    const std::vector<double> points = P.data(), c = Z.center().data();

    double worst_excess = -1e9, reached = 1e9;
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        const double rho = support(Z, d);
        double best = -1e300, centre = 0.0;
        for (int i = 0; i < n; ++i) centre += d[i] * c[i];
        for (int j = 0; j < N; ++j) best = std::max(best, dot_column(points, n, N, j, d));
        worst_excess = std::max(worst_excess, best - rho);
        reached = std::min(reached, (best - centre) / (rho - centre));
    }
    check(worst_excess <= 1e-10, b + ": a random point left the set");
    check(reached > 0.6, b + ": the points do not fill the set");

    std::vector<double> mean(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < N; ++j) mean[i] += points[i * N + j] / N;
        check(std::abs(mean[i] - c[i]) < 0.12, b + ": the points are not centred on the set");
    }
    check(!close(Z.rand_point(N, rng), P), b + ": successive draws differ");
    cora::Rng again(8);
    Zonotope::generate_random(n, m, again);
    check(close(Z.rand_point(N, again), P), b + ": the same seed gives the same points");
}

/// Extreme points are corners of the generator cube: `c + G b` with every `b_j = ±1`.
void extreme_points_are_corners(const std::string &b) {
    cora::Rng rng(9);
    const int n = 3, m = 4, N = 40;
    const Zonotope Z = Zonotope::generate_random(n, m, rng);
    const std::vector<double> points = Z.rand_point(N, rng, "extreme").data();
    const std::vector<double> c = Z.center().data(), G = Z.generators().data();
    double worst = 0.0;
    for (int j = 0; j < N; ++j) {
        double nearest = 1e300;
        for (int corner = 0; corner < (1 << m); ++corner) {
            double dist = 0.0;
            for (int i = 0; i < n; ++i) {
                double x = c[i];
                for (int k = 0; k < m; ++k) x += G[i * m + k] * ((corner >> k & 1) ? 1.0 : -1.0);
                dist += std::abs(points[i * N + j] - x);
            }
            nearest = std::min(nearest, dist);
        }
        worst = std::max(worst, nearest);
    }
    check(worst < 1e-10, b + ": an extreme point is not a corner");
}

/// A negative number of points is refused; with the checks off the draw is the same.
void checks_and_their_switch() {
    const Zonotope S = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    cora::Rng rng(3), again(3);
    check(test::raises([&] { S.rand_point(-1, rng); }, Error::wrong_value), "rand_point: N = -1");
    const Tensor on = S.rand_point(4, rng);
    ChecksScope off(false);
    check(test::close(S.rand_point(4, again), on, 0), "rand_point: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        rand_point_samples_the_set(b);
        extreme_points_are_corners(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope rand_point");
}
