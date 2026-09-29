// test_zonotope_linComb - zonotope linComb: the enclosure of the segments between two zonotopes

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "testing.h"

#include <algorithm>

using namespace cora::ct;
using test::check;
using test::close;
using test::column;
using test::random_direction;
using test::support;

namespace {

/// `linComb(Z, M Z + t)` has `2m + 1` generators and encloses both sets.
void linComb_encloses_both(const std::string &b) {
    cora::Rng rng(5);
    const int n = 3, m = 4;
    const Zonotope Z = Zonotope::generateRandom(n, m, rng);
    std::vector<double> M(n * n), t(n);
    rng.normal(M.data(), M.size(), 0.5);
    rng.normal(t.data(), t.size(), 1.0);
    const Zonotope mapped = Z.mtimes(Tensor::fromData(M, {n, n}));
    const Zonotope Z2(mapped.c + column(t), mapped.G);
    const Zonotope hull = Z.linComb(Z2);
    check(hull.G.shape() == std::vector<int64_t>({n, 2 * m + 1}), b + ": 2m + 1 generators");
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        check(support(hull, d) >= std::max(support(Z, d), support(Z2, d)) - 1e-10,
              b + ": linComb leaves one of its sets");
    }
    check(close(Z.linComb(Z).interval().inf, Z.interval().inf, 1e-12), b + ": linComb(Z, Z) is Z");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { linComb_encloses_both(b); });
    return test::finish("zonotope linComb");
}
