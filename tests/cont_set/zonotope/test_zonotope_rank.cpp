// test_zonotope_rank - zonotope rank: the rank of the generators
//
// CORA: contSet/@zonotope/rank.m
// PROPERTY: contSet/@zonotope/rank.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <tuple>

using namespace cora;
using test::check;
using test::column;

namespace {

/// CORA's unit test (a dependent and a zero column: rank 2), the identity, no generator, and the
/// tolerance: a singular value of 1e-9 counts for tol 1e-12 and not for tol 1e-8.
void rank_known(const std::string &b) {
    const Zonotope Z(column({1.0, 5.0, -1.0}),
                     Tensor({{2.0, 0.0, 4.0}, {6.0, 0.0, 0.0}, {4.0, 0.0, 8.0}}));
    check(Z.rank() == 2, b + ": CORA's example has rank 2");
    check(Zonotope(column({0.0, 0.0, 0.0}), Tensor::eye(3)).rank() == 3, b + ": the identity");
    check(Zonotope::origin(2).rank() == 0, b + ": no generators");
    const Zonotope thin(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1e-9}}));
    check(thin.rank(1e-12) == 2 && thin.rank(1e-8) == 1, b + ": the tolerance");
    check(thin.rank(0.0) == 2, b + ": tol 0 counts every nonzero singular value");
}

/// Stated property: generators G = A B with A (n, r) and B (r, m) of random entries have rank r,
/// and the rank is at most min(n, m); is_full_dim is rank == n.
void rank_of_products(const std::string &b) {
    cora::Rng rng(41);
    for (const auto [n, m, r] : {std::tuple{4, 6, 2}, {5, 5, 5}, {3, 7, 1}, {6, 3, 3}}) {
        const Zonotope A = Zonotope::generate_random(n, r, rng), B = Zonotope::generate_random(r, m, rng);
        const Zonotope Z(A.center(), A.generators().matmul(B.generators()));
        check(Z.rank() == r, b + ": rank " + std::to_string(r));
        check(Z.rank() <= std::min(n, m), b + ": at most min(n, m)");
        check(Z.is_full_dim() == (Z.rank() == n), b + ": full dimension is rank n");
    }
}

/// A batch and a negative tolerance are refused; with the checks off the rank is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { Zonotope::stack({Z, Z}).rank(); }, Error::not_supported), "rank: a batch");
    check(test::raises([&] { Z.rank(-1.0); }, Error::wrong_value), "rank: a negative tol");
    ChecksScope off(false);
    check(Z.rank() == 2, "rank: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        rank_known(b);
        rank_of_products(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope rank");
}
