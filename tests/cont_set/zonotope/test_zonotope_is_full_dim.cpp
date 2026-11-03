// test_zonotope_is_full_dim - zonotope is_full_dim: the generators span the space
//
// CORA: contSet/@zonotope/isFullDim.m
// PROPERTY: contSet/@zonotope/isFullDim.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: full rank, rank deficient, the tolerance, no generators.
void full_dim_known(const std::string &b) {
    const Zonotope full(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const Zonotope flat(column({0.0, 0.0}), Tensor({{1.0, 2.0}, {1.0, 2.0}}));
    check(full.is_full_dim(), b + ": the unit square is full-dimensional");
    check(!flat.is_full_dim(), b + ": parallel generators span a line");
    const Zonotope thin(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1e-7}}));
    check(!thin.is_full_dim(1e-6), b + ": a singular value 1e-7 is zero at CORA's tol 1e-6");
    check(thin.is_full_dim(), b + ": and nonzero at the repository tolerance");
    check(thin.is_full_dim(1e-8), b + ": and nonzero at tol 1e-8");
    check(!Zonotope(column({1.0}), Tensor::zeros({1, 0})).is_full_dim(), b + ": a point is not");

    const auto [res, basis] = flat.is_full_dim_subspace();
    check(!res, b + ": subspace: not full");
    check(basis.shape() == std::vector<int64_t>({2, 1}), b + ": a basis of the line");
    check(close(basis.abs(), std::vector<double>{std::sqrt(0.5), std::sqrt(0.5)}, 1e-12),
          b + ": the direction (1, 1) / sqrt(2)");
    const auto [res2, identity] = full.is_full_dim_subspace();
    check(res2 && close(identity, std::vector<double>{1, 0, 0, 1}, 0), b + ": full: the identity");
    check(Zonotope(column({1.0}), Tensor::zeros({1, 0})).is_full_dim_subspace().second.shape() ==
              std::vector<int64_t>({1, 0}),
          b + ": a point has an empty basis");
}

/// Stated property: a random zonotope with m >= n generators is full-dimensional, and appending
/// a combination of generators to a rank-deficient set does not change the span.
void full_dim_property(const std::string &b) {
    cora::Rng rng(8);
    check(Zonotope::generate_random(4, 6, rng).is_full_dim(), b + ": random 4 x 6");
    const Zonotope Z = Zonotope::generate_random(4, 3, rng); // 3 generators in 4 dimensions
    check(!Z.is_full_dim(), b + ": fewer generators than dimensions");
    const auto [res, basis] = Z.is_full_dim_subspace();
    check(!res && basis.shape() == std::vector<int64_t>({4, 3}), b + ": a basis of 3 vectors");
    // every generator lies in the span of the basis: (I - B B') G = 0
    const Tensor residual =
        Z.generators() - basis.matmul(basis.matrix_transpose().matmul(Z.generators()));
    check(residual.abs().max().item() < 1e-12, b + ": the generators lie in the span");
}

/// A batch and a negative tol are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope full(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { Zonotope::stack({full, full}).is_full_dim(); }, Error::not_supported),
          "is_full_dim: a batch");
    check(test::raises([&] { full.is_full_dim(-1.0); }, Error::wrong_value),
          "is_full_dim: tol < 0");
    ChecksScope off(false);
    check(full.is_full_dim(), "is_full_dim: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        full_dim_known(b);
        full_dim_property(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope is_full_dim");
}
