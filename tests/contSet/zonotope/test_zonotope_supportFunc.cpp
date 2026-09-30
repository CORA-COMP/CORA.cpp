// test_zonotope_supportFunc - zonotope supportFunc: the closed form against the best corner of the
// generator cube

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "testing.h"

#include <algorithm>

using namespace cora;
using test::check;
using test::close;
using test::random_direction;
using test::support;

namespace {

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

void supportFunc_matches_brute_force(const std::string &b) {
    cora::Rng rng(1);
    for (const auto [n, m] : {std::pair{1, 2}, {2, 3}, {3, 5}, {4, 6}}) {
        const Zonotope Z = Zonotope::generateRandom(n, m, rng);
        for (int i = 0; i < 5; ++i) {
            const std::vector<double> d = random_direction(rng, n);
            check(close(support(Z, d), brute_support(Z, d), 1e-10),
                  b + ": supportFunc at " + std::to_string(n) + "x" + std::to_string(m));
        }
    }
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { supportFunc_matches_brute_force(b); });
    return test::finish("zonotope supportFunc");
}
