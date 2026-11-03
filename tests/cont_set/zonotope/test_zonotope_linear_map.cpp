// test_zonotope_linear_map - zonotope linear_map: by a matrix (exact) and by an interval matrix (an
// enclosure)
//
// CORA: contSet/@zonotope/mtimes.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::random_direction;
using test::support;

namespace {

/// `ρ(d, M Z) = ρ(Mᵀd, Z)`.
void mtimes_by_a_matrix(const std::string &b) {
    cora::Rng rng(2);
    const int n = 3, m = 4;
    const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, m);
    std::vector<double> M(n * n);
    rng.normal(M.data(), M.size(), 1.0);
    const Zonotope MZ = Z.linear_map(Tensor::from_data(M, {n, n}));
    check(MZ.generators().shape() == std::vector<int64_t>({n, m}),
          b + ": linear_map keeps the generators");
    for (int k = 0; k < 5; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        std::vector<double> Mtd(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) Mtd[j] += M[i * n + j] * d[i];
        check(close(support(MZ, d), support(Z, Mtd), 1e-10), b + ": linear_map by a matrix");
    }
}

/// The interval matrix with no width is a matrix; with width it encloses every matrix in it.
void mtimes_by_an_interval_matrix(const std::string &b) {
    cora::Rng rng(3);
    const int n = 3, m = 4;
    const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, m);
    std::vector<double> centre(n * n), radius(n * n);
    rng.normal(centre.data(), centre.size(), 1.0);
    rng.uniform(radius.data(), radius.size(), 0.0, 0.3);
    const Tensor C = Tensor::from_data(centre, {n, n}), R = Tensor::from_data(radius, {n, n});

    const Zonotope point = Z.linear_map(Interval(C, C));
    const Zonotope plain = Z.linear_map(C);
    check(close(point.center(), plain.center()) &&
              close(point.generators().data().size(), plain.generators().data().size() + n * n),
          b + ": a degenerate interval matrix adds only zero generators");
    for (int k = 0; k < 5; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        check(close(support(point, d), support(plain, d), 1e-10), b + ": no width, no enlargement");
    }

    const Zonotope wide = Z.linear_map(Interval(C - R, C + R));
    check(wide.generators().shape() == std::vector<int64_t>({n, m + n}),
          b + ": one generator per dimension");
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> M = centre, u(n * n);
        rng.uniform(u.data(), u.size(), -1.0, 1.0);
        for (int i = 0; i < n * n; ++i) M[i] += radius[i] * u[i];
        const Zonotope image = Z.linear_map(Tensor::from_data(M, {n, n}));
        for (int k = 0; k < 3; ++k) {
            const std::vector<double> d = random_direction(rng, n);
            check(support(wide, d) >= support(image, d) - 1e-10,
                  b + ": an interval matrix's product misses a member's");
        }
    }
}

/// A matrix of the wrong width is refused; with the checks off the map is the same.
void checks_and_their_switch() {
    const Zonotope S = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    const Tensor M({{1.0, 2.0}, {0.0, 1.0}}), wide({{1.0, 2.0, 3.0}});
    check(test::raises([&] { S.linear_map(wide); }, Error::dimension_mismatch),
          "linear_map: a matrix of three columns");
    const Tensor on = S.linear_map(M).center();
    ChecksScope off(false);
    check(test::close(S.linear_map(M).center(), on, 0), "linear_map: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        mtimes_by_a_matrix(b);
        mtimes_by_an_interval_matrix(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope linear_map");
}
