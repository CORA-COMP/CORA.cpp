// test_zonotope_mtimes - zonotope mtimes: by a matrix (exact) and by an interval matrix (an
// enclosure)

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
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
    const Zonotope Z = Zonotope::generateRandom(n, m, rng);
    std::vector<double> M(n * n);
    rng.normal(M.data(), M.size(), 1.0);
    const Zonotope MZ = Z.mtimes(Tensor::fromData(M, {n, n}));
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
    const Zonotope Z = Zonotope::generateRandom(n, m, rng);
    std::vector<double> centre(n * n), radius(n * n);
    rng.normal(centre.data(), centre.size(), 1.0);
    rng.uniform(radius.data(), radius.size(), 0.0, 0.3);
    const Tensor C = Tensor::fromData(centre, {n, n}), R = Tensor::fromData(radius, {n, n});

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
        const Zonotope image = Z.mtimes(Tensor::fromData(M, {n, n}));
        for (int k = 0; k < 3; ++k) {
            const std::vector<double> d = random_direction(rng, n);
            check(support(wide, d) >= support(image, d) - 1e-10,
                  b + ": an interval matrix's product misses a member's");
        }
    }
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        mtimes_by_a_matrix(b);
        mtimes_by_an_interval_matrix(b);
    });
    return test::finish("zonotope mtimes");
}
