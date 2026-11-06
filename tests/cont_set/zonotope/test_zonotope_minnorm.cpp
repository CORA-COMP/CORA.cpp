// test_zonotope_minnorm - zonotope minnorm: the distance from the center to the boundary
//
// CORA: contSet/@zonotope/minnorm.m
// PROPERTY: contSet/@zonotope/minnorm.m
// CORA authors: Victor Gassmann
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the square [-1, 1]^2 (first facet on a tie), a rectangle, a diamond, a flat set.
void minnorm_known(const std::string &b) {
    const auto [v, x] = Zonotope(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})).minnorm();
    check(close(v, std::vector<double>{1}, 1e-14), b + ": the square: distance 1");
    check(close(x, std::vector<double>{0, -1}, 1e-14),
          b + ": the point of the first facet, (0, -1)");
    const auto [v2, x2] = Zonotope(column({1.0, 1.0}), Tensor({{2.0, 0.0}, {0.0, 1.0}})).minnorm();
    check(close(v2, std::vector<double>{1}, 1e-14) && close(x2, std::vector<double>{1, 0}, 1e-14),
          b + ": the rectangle about (1, 1): distance 1 to the lower edge");
    const auto [v3, x3] = Zonotope(column({0.0, 0.0}), Tensor({{1.0, 1.0}, {1.0, -1.0}})).minnorm();
    check(close(v3, std::vector<double>{std::sqrt(2.0)}, 1e-14),
          b + ": the diamond |x| + |y| <= 2");
    check(close(x3.matrix_transpose().matmul(x3), std::vector<double>{2}, 1e-12),
          b + ": the nearest point is at distance sqrt(2)");
    const auto [v4, x4] = Zonotope(column({3.0, 4.0}), Tensor({{1.0}, {1.0}})).minnorm();
    check(close(v4, std::vector<double>{0}, 0) && close(x4, std::vector<double>{3, 4}, 0),
          b + ": a flat zonotope: distance 0 at the center");
}

/// Stated property: the ball of radius minnorm about the center is in the zonotope (the support
/// function exceeds the center's by at least minnorm), the point is on the boundary at that
/// distance.
void minnorm_is_the_inradius(const std::string &b) {
    cora::Rng rng(17);
    for (const auto &[n, m] : {std::pair{2, 4}, {3, 5}}) {
        const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, m);
        const auto [value, x] = Z.minnorm();
        const double val = value.item();
        double least = 1e300;
        for (int k = 0; k < 400; ++k) {
            std::vector<double> d = test::random_direction(rng, n);
            double len = 0, shift = 0;
            for (int i = 0; i < n; ++i) len += d[i] * d[i];
            len = std::sqrt(len);
            for (int i = 0; i < n; ++i) shift += d[i] / len * Z.center().data()[i];
            const double reach = test::support(Z, d) / len - shift;
            check(reach >= val - 1e-9, b + ": the ball of radius minnorm leaves the zonotope");
            least = std::min(least, reach);
        }
        check(least <= val * 1.5 + 1e-9, b + ": sampled directions come close to the inradius");
        double distance = 0;
        for (int i = 0; i < n; ++i) distance += std::pow(x.data()[i] - Z.center().data()[i], 2);
        // the point lies at the distance
        check(close(std::sqrt(distance), val, 1e-9), b + ": the point lies at distance minnorm");
        check(Z.contains(x, "exact", 1e-9), b + ": and in the zonotope");
    }
}

/// A batch is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { Zonotope::stack({Z, Z}).minnorm(); }, Error::not_supported),
          "minnorm: a batch");
    const Tensor on = Z.minnorm().first;
    ChecksScope off(false);
    check(close(Z.minnorm().first, on, 0), "minnorm: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        minnorm_known(b);
        minnorm_is_the_inradius(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope minnorm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
