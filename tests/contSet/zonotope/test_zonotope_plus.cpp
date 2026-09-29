// test_zonotope_plus - zonotope plus: the Minkowski sum

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;
using test::random_direction;
using test::support;

namespace {

void plus_is_the_minkowski_sum(const std::string &b) {
    cora::Rng rng(4);
    const int n = 3;
    const Zonotope A = Zonotope::generateRandom(n, 3, rng), B = Zonotope::generateRandom(n, 2, rng);
    const Zonotope S = A.plus(B);
    check(S.G.shape() == std::vector<int64_t>({n, 5}), b + ": generators are joined");
    check(close(S.c, A.c + B.c), b + ": centres add");
    for (int k = 0; k < 5; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        check(close(support(S, d), support(A, d) + support(B, d), 1e-10), b + ": supports add");
    }
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { plus_is_the_minkowski_sum(b); });
    return test::finish("zonotope plus");
}
