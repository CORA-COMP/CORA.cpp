// The zonotope on every backend: each operation against its definition, checked with
// the support function, which for a zonotope has a closed form and a brute-force twin (the
// maximum over the corners of the generator cube).

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;

namespace {

Tensor column(const std::vector<double> &v) { return Tensor::from_data(v, {int64_t(v.size()), 1}); }

/// `max_b dᵀ(c + G b)` over the corners `b ∈ {-1, 1}^m`, on the host.
double brute_support(const Zonotope &Z, const std::vector<double> &d) {
    const int n = Z.dim();
    const std::vector<double> c = Z.c.data(), G = Z.G.data();
    const int m = static_cast<int>(G.size()) / n;
    double best = -1e300;
    for (int corner = 0; corner < (1 << m); ++corner) {
        double value = 0.0;
        for (int i = 0; i < n; ++i) {
            double x = c[i];
            for (int j = 0; j < m; ++j) x += G[i * m + j] * ((corner >> j & 1) ? 1.0 : -1.0);
            value += d[i] * x;
        }
        best = std::max(best, value);
    }
    return best;
}

std::vector<double> random_direction(cora::Rng &rng, int n) {
    std::vector<double> d(n);
    rng.normal(d.data(), d.size(), 1.0);
    return d;
}

double support(const Zonotope &Z, const std::vector<double> &d) {
    return Z.support_func(column(d)).data()[0];
}

void basics(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0, 0.5}, {0.0, 2.0, 0.5}}));
    check(Z.dim() == 2, b + ": dim");
    check(close(Z.center(), std::vector<double>{1, 2}), b + ": center");
    check(close(Z.interval().inf, std::vector<double>{-0.5, -0.5}), b + ": the hull's lower corner");
    check(close(Z.interval().sup, std::vector<double>{2.5, 4.5}), b + ": the hull's upper corner");
}

void support_func_matches_brute_force(const std::string &b) {
    cora::Rng rng(1);
    for (const auto [n, m] : {std::pair{1, 2}, {2, 3}, {3, 5}, {4, 6}}) {
        const Zonotope Z = Zonotope::generate_random(n, m, rng);
        for (int i = 0; i < 5; ++i) {
            const std::vector<double> d = random_direction(rng, n);
            check(close(support(Z, d), brute_support(Z, d), 1e-10),
                  b + ": supportFunc at " + std::to_string(n) + "x" + std::to_string(m));
        }
    }
}

/// `ρ(d, M Z) = ρ(Mᵀd, Z)`.
void mtimes_by_a_matrix(const std::string &b) {
    cora::Rng rng(2);
    const int n = 3, m = 4;
    const Zonotope Z = Zonotope::generate_random(n, m, rng);
    std::vector<double> M(n * n);
    rng.normal(M.data(), M.size(), 1.0);
    const Zonotope MZ = Z.mtimes(Tensor::from_data(M, {n, n}));
    check(MZ.G.shape() == std::vector<int64_t>({n, m}), b + ": mtimes keeps the generators");
    for (int k = 0; k < 5; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        std::vector<double> Mtd(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) Mtd[j] += M[i * n + j] * d[i];
        check(close(support(MZ, d), support(Z, Mtd), 1e-10), b + ": mtimes by a matrix");
    }
}

/// The interval matrix with no width is a matrix; with width it encloses every matrix in it.
void mtimes_by_an_interval_matrix(const std::string &b) {
    cora::Rng rng(3);
    const int n = 3, m = 4;
    const Zonotope Z = Zonotope::generate_random(n, m, rng);
    std::vector<double> centre(n * n), radius(n * n);
    rng.normal(centre.data(), centre.size(), 1.0);
    rng.uniform(radius.data(), radius.size(), 0.0, 0.3);
    const Tensor C = Tensor::from_data(centre, {n, n}), R = Tensor::from_data(radius, {n, n});

    const Zonotope point = Z.mtimes(Interval(C, C));
    const Zonotope plain = Z.mtimes(C);
    check(close(point.c, plain.c) && close(point.G.data().size(), plain.G.data().size() + n * n),
          b + ": a degenerate interval matrix adds only zero generators");
    for (int k = 0; k < 5; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        check(close(support(point, d), support(plain, d), 1e-10), b + ": no width, no enlargement");
    }

    const Zonotope wide = Z.mtimes(Interval(C - R, C + R));
    check(wide.G.shape() == std::vector<int64_t>({n, m + n}), b + ": one generator per dimension");
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> M = centre, u(n * n);
        rng.uniform(u.data(), u.size(), -1.0, 1.0);
        for (int i = 0; i < n * n; ++i) M[i] += radius[i] * u[i];
        const Zonotope image = Z.mtimes(Tensor::from_data(M, {n, n}));
        for (int k = 0; k < 3; ++k) {
            const std::vector<double> d = random_direction(rng, n);
            check(support(wide, d) >= support(image, d) - 1e-10,
                  b + ": an interval matrix's product misses a member's");
        }
    }
}

void plus_is_the_minkowski_sum(const std::string &b) {
    cora::Rng rng(4);
    const int n = 3;
    const Zonotope A = Zonotope::generate_random(n, 3, rng), B = Zonotope::generate_random(n, 2, rng);
    const Zonotope S = A.plus(B);
    check(S.G.shape() == std::vector<int64_t>({n, 5}), b + ": generators are joined");
    check(close(S.c, A.c + B.c), b + ": centres add");
    for (int k = 0; k < 5; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        check(close(support(S, d), support(A, d) + support(B, d), 1e-10), b + ": supports add");
    }
}

/// `linComb(Z, M Z + t)` has `2m + 1` generators and encloses both sets.
void lin_comb_encloses_both(const std::string &b) {
    cora::Rng rng(5);
    const int n = 3, m = 4;
    const Zonotope Z = Zonotope::generate_random(n, m, rng);
    std::vector<double> M(n * n), t(n);
    rng.normal(M.data(), M.size(), 0.5);
    rng.normal(t.data(), t.size(), 1.0);
    const Zonotope mapped = Z.mtimes(Tensor::from_data(M, {n, n}));
    const Zonotope Z2(mapped.c + column(t), mapped.G);
    const Zonotope hull = Z.lin_comb(Z2);
    check(hull.G.shape() == std::vector<int64_t>({n, 2 * m + 1}), b + ": 2m + 1 generators");
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        check(support(hull, d) >= std::max(support(Z, d), support(Z2, d)) - 1e-10,
              b + ": linComb leaves one of its sets");
    }
    check(close(Z.lin_comb(Z).interval().inf, Z.interval().inf, 1e-12), b + ": linComb(Z, Z) is Z");
}

void interval_is_the_hull(const std::string &b) {
    cora::Rng rng(6);
    const Zonotope Z = Zonotope::generate_random(4, 6, rng);
    const Interval I = Z.interval();
    for (int i = 0; i < 4; ++i) {
        std::vector<double> e(4, 0.0), f(4, 0.0);
        e[i] = 1.0;
        f[i] = -1.0;
        check(close(I.support_func(column(e)).data()[0], support(Z, e), 1e-12), b + ": hull, upper");
        check(close(I.support_func(column(f)).data()[0], support(Z, f), 1e-12), b + ": hull, lower");
    }
    for (int k = 0; k < 10; ++k) {
        const std::vector<double> d = random_direction(rng, 4);
        check(I.support_func(column(d)).data()[0] >= support(Z, d) - 1e-10, b + ": the hull contains Z");
    }
}

void generate_random_follows_cora(const std::string &b) {
    cora::Rng rng(7);
    const Zonotope Z = Zonotope::generate_random(5, 8, rng);
    check(Z.c.shape() == std::vector<int64_t>({5, 1}) && Z.G.shape() == std::vector<int64_t>({5, 8}),
          b + ": shapes");
    const std::vector<double> G = Z.G.data();
    bool short_enough = true;
    for (int j = 0; j < 8; ++j) {
        double norm = 0.0;
        for (int i = 0; i < 5; ++i) norm += G[i * 8 + j] * G[i * 8 + j];
        short_enough &= std::sqrt(norm) <= 1.0 + 1e-12;
    }
    check(short_enough, b + ": generators are at most unit length");
    check(!close(Zonotope::generate_random(5, 8, rng).c, Z.c), b + ": successive sets differ");
}

double dot_column(const std::vector<double> &points, int n, int N, int j, const std::vector<double> &d) {
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
    const std::vector<double> points = P.data(), c = Z.c.data();

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
    const std::vector<double> points = Z.rand_point(N, rng, true).data();
    const std::vector<double> c = Z.c.data(), G = Z.G.data();
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

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        basics(b);
        support_func_matches_brute_force(b);
        mtimes_by_a_matrix(b);
        mtimes_by_an_interval_matrix(b);
        plus_is_the_minkowski_sum(b);
        lin_comb_encloses_both(b);
        interval_is_the_hull(b);
        generate_random_follows_cora(b);
        rand_point_samples_the_set(b);
        extreme_points_are_corners(b);
    });
    return test::finish("zonotope");
}
