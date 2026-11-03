// test_zonotope_zonotope_norm - zonotope zonotope_norm: the norm with the zonotope as unit ball
//
// CORA: contSet/@zonotope/zonotopeNorm.m
// PROPERTY: contSet/@zonotope/zonotopeNorm.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// CORA's unit test: G = [1 -2 2 0; -1 1 0 1] gives 2.2 at (5, 3) and 1 at (-5, 3); the center
/// does not matter; a point has norm 0 at the origin and infinity elsewhere.
void cora_unit_test(const std::string &b) {
    const Tensor G({{1.0, -2.0, 2.0, 0.0}, {-1.0, 1.0, 0.0, 1.0}});
    const Zonotope Z(column({0.0, 0.0}), G);
    check(close(Z.zonotope_norm(column({5.0, 3.0})), std::vector<double>{2.2}, 1e-9), b + ": 2.2");
    check(close(Z.zonotope_norm(column({-5.0, 3.0})), std::vector<double>{1}, 1e-9), b + ": 1");
    const Zonotope shifted(column({1.0, -2.0}), G);
    check(close(shifted.zonotope_norm(column({5.0, 3.0})), std::vector<double>{2.2}, 1e-9),
          b + ": the center is ignored");
    const Zonotope point(column({0.0, 0.0}), Tensor::zeros({2, 0}));
    check(close(point.zonotope_norm(column({0.0, 0.0})), std::vector<double>{0}, 0), b + ": a point: 0");
    check(point.zonotope_norm(column({1.0, 0.0})).item() == kInf, b + ": a point: infinity");
    const Zonotope line(column({0.0, 0.0}), Tensor({{1.0}, {0.0}}));
    check(line.zonotope_norm(column({0.0, 1.0})).item() == kInf, b + ": outside the span: infinity");
    check(line.zonotope_norm_minimizer(column({0.0, 1.0})).second.shape() == Shape{0, 1},
          b + ": no minimizer then");
}

/// Stated property: the minimizer reaches the point (G beta = p, |beta|_inf = the norm), and the
/// boundary point of the support function in a direction has norm 1.
void the_minimizer_reaches_the_point(const std::string &b) {
    cora::Rng rng(71);
    const Zonotope Z(column({0.0, 0.0, 0.0}), Zonotope::generate_random(rng, 3, std::nullopt, 5).generators());
    for (int k = 0; k < 10; ++k) {
        const Tensor p = column(test::random_direction(rng, 3));
        const auto [value, beta] = Z.zonotope_norm_minimizer(p);
        check(beta.shape() == Shape{5, 1}, b + ": minimizer shape");
        check(close(Z.generators().matmul(beta), p, 1e-8), b + ": G beta = p");
        check(close(beta.abs().max().item(), value.item(), 1e-8), b + ": |beta|_inf is the norm");
        // the point p / norm lies on the boundary: contained, and a larger multiple is not
        check(Z.contains(p / value.item(), "exact", 1e-7), b + ": p / norm is in Z");
        check(!Z.contains(p / (value.item() * 0.98), "exact", 1e-9), b + ": and on its boundary");
    }
}

/// A shifted zonotope warns when the minimizer is asked for; a batch, a point of the wrong shape
/// are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { Zonotope::stack({Z, Z}).zonotope_norm(column({1.0, 0.0})); },
                       Error::not_supported), "zonotope_norm: a batch");
    check(test::raises([&] { Z.zonotope_norm(column({1.0, 0.0, 0.0})); }, Error::dimension_mismatch),
          "zonotope_norm: a point of dimension 3");
    int warnings = 0;
    const WarningHandler previous =
        set_warning_handler([&](const std::string &, const std::string &) { ++warnings; });
    Zonotope(column({1.0, 0.0}), Tensor({{1.0}, {0.0}})).zonotope_norm_minimizer(column({1.0, 0.0}));
    Z.zonotope_norm_minimizer(column({1.0, 0.0}));
    set_warning_handler(previous);
    check(warnings == 1, "zonotope_norm_minimizer: one warning, for the shifted zonotope");
    const Tensor on = Z.zonotope_norm(column({0.5, 2.0}));
    ChecksScope off(false);
    check(close(Z.zonotope_norm(column({0.5, 2.0})), on, 0), "zonotope_norm: same with checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        cora_unit_test(b);
        the_minimizer_reaches_the_point(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope zonotope_norm");
}
