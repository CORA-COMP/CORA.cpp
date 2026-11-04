// test_zonotope_abs - zonotope abs: the absolute value of center and generators
//
// CORA: contSet/@zonotope/abs.m
// PROPERTY: contSet/@zonotope/abs.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// A hand-computed case, a batch member by member, and the zonotope of a negated one.
void abs_of_known_zonotopes(const std::string &b) {
    const Zonotope Z(column({1.0, -2.0}), Tensor({{1.0, -3.0}, {-2.0, 0.5}}));
    const Zonotope A = Z.abs();
    check(close(A.center(), std::vector<double>{1, 2}, 0), b + ": |c|");
    check(close(A.generators(), std::vector<double>{1, 3, 2, 0.5}, 0), b + ": |G|");
    const Zonotope batch = Zonotope::stack({Z, -Z});
    check(close(batch.abs().generators().slice(0, 1, 2).squeeze(0), A.generators(), 0),
          b + ": a batch member");
}

/// Stated property: the corner |c| + sum |g_j| of abs(Z) bounds |x| for every x in Z, abs is
/// and idempotent.
void abs_corner_bounds_the_set(const std::string &b) {
    cora::Rng rng(11);
    const int n = 4;
    const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, 6);
    const Zonotope A = Z.abs();
    const Tensor corner = A.center() + A.generators().abs_sum();
    const Tensor x = Z.rand_point(200, rng);
    check((x.abs() - corner).max().item() <= 1e-12, b + ": |x| <= |c| + sum |g_j|");
    check(close(A.abs().generators(), A.generators(), 0), b + ": idempotent");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        abs_of_known_zonotopes(b);
        abs_corner_bounds_the_set(b);
    });
    ChecksScope off(false);
    const Zonotope Z(column({1.0, -2.0}), Tensor({{1.0, -3.0}, {-2.0, 0.5}}));
    check(close(Z.abs().center(), std::vector<double>{1, 2}, 0),
          "abs: the same with the checks off");
    return test::finish("zonotope abs");
}
