// test_poly_zonotope_split_dep_factor - PolyZonotope split_dep_factor: the halves of a factor
//
// A factor alpha in [-1, 1] is halved by alpha = (beta + 1) / 2 and alpha = (beta - 1) / 2: the
// first half at beta is the original set at (beta + 1) / 2, the second at (beta - 1) / 2. A hand
// case with the exponents 1 and 2, the stated property on random sets, and the arguments.
//
// CORA: contSet/@polyZonotope/splitDepFactor.m
// PROPERTY: contSet/@polyZonotope/splitDepFactor.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The point of the dependent part at the factors alpha (one per identifier).
std::vector<double> aux_point(const PolyZonotope &P, const std::vector<double> &alpha) {
    const int64_t n = P.dim(), h = P.num_dependent();
    std::vector<double> p = P.center().data();
    const std::vector<double> G = P.generators().data();
    for (int64_t j = 0; j < h; ++j) {
        double mono = 1;
        for (std::size_t r = 0; r < alpha.size(); ++r)
            mono *= std::pow(alpha[r], P.exponents()[std::size_t(j)][r]);
        for (int64_t i = 0; i < n; ++i) p[std::size_t(i)] += G[std::size_t(i * h + j)] * mono;
    }
    return p;
}

/// c = 1, 2 a + 3 a^2: the halves are 2.75 + 2.5 b + 0.75 b^2 and 0.75 - 0.5 b + 0.75 b^2.
PolyZonotope aux_known() {
    return {
        column({1.0}), Tensor::from_data({2, 3}, {1, 2}), Tensor::zeros({1, 0}), {{1}, {2}}, {1}};
}

/// Stated property: the halves are the set on [0, 1] and [-1, 0] of the factor.
void split_property(const std::string &b) {
    Rng rng(17);
    for (int k = 0; k < 10; ++k) {
        const PolyZonotope P = test::random_poly(rng, 2);
        const auto [P1, P2] = P.split_dep_factor(2);
        std::vector<double> alpha(2), beta(2);
        rng.uniform(beta.data(), 2, -1.0, 1.0);
        // the factor 2 is halved, the factor 1 stays
        alpha = {beta[0], (beta[1] + 1) / 2};
        check(close(Tensor::from_data(aux_point(P1, beta), {2, 1}),
                    Tensor::from_data(aux_point(P, alpha), {2, 1}), 1e-12),
              b + ": the first half is the set on [0, 1]");
        alpha = {beta[0], (beta[1] - 1) / 2};
        check(close(Tensor::from_data(aux_point(P2, beta), {2, 1}),
                    Tensor::from_data(aux_point(P, alpha), {2, 1}), 1e-12),
              b + ": the second half is the set on [-1, 0]");
        check(P1.ids() == P.ids(), b + ": the identifiers stay");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const auto [P1, P2] = aux_known().split_dep_factor(1);
        check(close(P1.center(), column({2.75})) &&
                  close(P1.generators(), Tensor::from_data({2.5, 0.75}, {1, 2})),
              b + ": the first half by hand");
        check(close(P2.center(), column({0.75})) &&
                  close(P2.generators(), Tensor::from_data({-0.5, 0.75}, {1, 2})),
              b + ": the second half by hand");
        split_property(b);
    });
    check(test::raises([] { aux_known().split_dep_factor(5); }, Error::wrong_value),
          "split_dep_factor: an unknown identifier");
    ChecksScope off(false);
    check(aux_known().split_dep_factor(1).first.dim() == 1, "split_dep_factor: valid, checks off");
    return test::finish("poly_zonotope split_dep_factor");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
