// test_con_zonotope_is_full_dim - ConZonotope::is_full_dim: the interior of a constrained zonotope
//
// CORA: contSet/@conZonotope/isFullDim.m
// PROPERTY: contSet/@conZonotope/isFullDim.m

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: with and without constraints, a flat set, a point pinned by the box, no point.
void full_dim_known(const std::string &b) {
    const ConZonotope plain(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(plain.is_full_dim(), b + ": the unit square");
    check(!ConZonotope(column({0.0, 0.0}), Tensor({{1.0, 2.0}, {1.0, 2.0}})).is_full_dim(),
          b + ": parallel generators span a line");
    // the third factor is fixed to zero: the square stays full-dimensional
    const ConZonotope square(column({0.0, 0.0}), Tensor({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}),
                             Tensor({{0.0, 0.0, 1.0}}), column({0.0}));
    check(square.is_full_dim(), b + ": a fixed factor does not matter");
    // the segment x = xi on xi1 + xi2 = 1 in the plane
    const ConZonotope segment(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), column({1.0}));
    check(!segment.is_full_dim(), b + ": a segment in the plane");
    // a segment on a line: x = xi1 + 2 xi2 with xi1 = xi2 has the interval [-3, 3]
    const ConZonotope line(column({0.0}), Tensor({{1.0, 2.0}}), Tensor({{1.0, -1.0}}),
                           column({0.0}));
    check(line.is_full_dim(), b + ": a segment in a line is full-dimensional");
    // CORA's rank test says true: x = xi1 + 2 xi2 with xi1 + xi2 = 2 is the point 3
    const ConZonotope pinned(column({0.0}), Tensor({{1.0, 2.0}}), Tensor({{1.0, 1.0}}),
                             column({2.0}));
    check(!pinned.is_full_dim(), b + ": the box pins it to the point 3 (CORA: true)");
    const ConZonotope none(column({0.0}), Tensor({{1.0, 2.0}}), Tensor({{1.0, 0.0}}),
                           column({5.0}));
    check(!none.is_full_dim(), b + ": a set without a point has no interior (CORA: true)");

    const auto [res, basis] = segment.is_full_dim_subspace();
    check(!res && basis.shape() == Shape{2, 1}, b + ": subspace: a basis of the line");
    check(close(basis.abs(), {std::sqrt(0.5), std::sqrt(0.5)}, 1e-9),
          b + ": the direction (1, -1)");
    const auto [full, identity] = square.is_full_dim_subspace();
    check(full && close(identity, {1, 0, 0, 1}, 0), b + ": full: the identity");
    check(none.is_full_dim_subspace().second.shape() == Shape{1, 0}, b + ": empty: no basis");
}

/// Stated property: a random set with at least n free factors is full-dimensional, and its span
/// agrees with the subspace that the second form returns.
void full_dim_property(const std::string &b) {
    cora::Rng rng(23);
    for (int k = 0; k < 6; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 6, 2);
        check(cz.is_full_dim(), b + ": random 3 x 6 with two constraints");
        const ConZonotope thin = test::random_con_zonotope(rng, 4, 4, 2);
        const auto [res, basis] = thin.is_full_dim_subspace();
        check(!res && basis.shape() == Shape{4, 2}, b + ": two free factors span two directions");
    }
}

/// A batch with constraints and a negative tol are refused; the same valid call with the checks
/// off.
void checks_and_their_switch() {
    const ConZonotope batch(Tensor::zeros({2, 1, 1}), Tensor::ones({2, 1, 2}));
    check(test::raises([&] { batch.is_full_dim(); }, Error::not_supported), "is_full_dim: a batch");
    check(test::raises([&] { batch.is_full_dim(-1.0); }, Error::wrong_value),
          "is_full_dim: tol < 0");
    const ConZonotope segment(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), column({1.0}));
    ChecksScope off(false);
    check(!segment.is_full_dim(), "is_full_dim: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        full_dim_known(b);
        full_dim_property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope is_full_dim");
}
