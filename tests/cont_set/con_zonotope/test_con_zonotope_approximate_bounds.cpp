// test_con_zonotope_approximate_bounds - ConZonotope::approximate_bounds: bounds of a batch
//
// CORA: contSet/@conZonotope/approximateBoundsWithGPU.m
// PROPERTY: contSet/@conZonotope/approximateBoundsWithGPU.m
// CORA authors: Lukas Koller
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

using Options = ConZonotope::BoundsOptions;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: x1 + x2 <= -3/2 on [-1, 1]^2 bounds both factors above by -1/2, so the box of
/// the identity generators is [-1, -1/2]^2; no constraint gives the zonotope's box.
void bounds_known(const std::string &b) {
    const Tensor c = Tensor::zeros({1, 2, 1}), G = Tensor::from_data({1, 0, 0, 1}, {1, 2, 2});
    const Tensor A = Tensor::from_data({1, 1}, {1, 1, 2}),
                 rhs = Tensor::from_data({-1.5}, {1, 1, 1});
    const ConZonotope::Bounds r = ConZonotope::approximate_bounds(c, G, A, rhs);
    check(close(r.l, {-1, -1}, 1e-12) && close(r.u, {-0.5, -0.5}, 1e-12),
          b + ": the box [-1, -1/2]^2");
    check(close(r.bl, {-1, -1}, 0) && close(r.bu, {-0.5, -0.5}, 0), b + ": the factor bounds");
    const ConZonotope::Bounds none =
        ConZonotope::approximate_bounds(c, G, Tensor::zeros({1, 0, 2}), Tensor::zeros({1, 0, 1}));
    check(close(none.l, {-1, -1}, 0) && close(none.u, {1, 1}, 0) && close(none.bu, {1, 1}, 0),
          b + ": without constraints the box of the zonotope");
    // the interval error widens the box
    const ConZonotope::Bounds wide =
        ConZonotope::approximate_bounds(c, G, A, rhs, c.full_like(c.shape(), 0.25));
    check(close(wide.l, {-1.25, -1.25}, 1e-12), b + ": dr widens it");
    // the exact method: the same box here, a linear program per bound
    Options exact;
    exact.method = "exact";
    const ConZonotope::Bounds e =
        ConZonotope::approximate_bounds(c, G, A, rhs, std::nullopt, 1, exact);
    check(close(e.l, {-1, -1}, 1e-9) && close(e.u, {-0.5, -0.5}, 1e-9),
          b + ": exact: the same box");
    // no point: x1 >= 1/2 and x1 <= -1/2 gives crossing bounds, NaN and 0
    const Tensor A2 = Tensor::from_data({-1, 0, 1, 0}, {1, 2, 2}),
                 rhs2 = Tensor::from_data({-0.5, -0.5}, {1, 2, 1});
    const ConZonotope::Bounds empty = ConZonotope::approximate_bounds(c, G, A2, rhs2);
    check(std::isnan(empty.l.data()[0]) && close(empty.bl, {0, 0}, 0), b + ": no point: NaN and 0");
    // a union of two rows: x1 >= 1/2 or x1 <= -1/2 gives the whole box in x1
    const ConZonotope::Bounds u = ConZonotope::approximate_bounds(c, G, A2, rhs2, std::nullopt, 2);
    check(close(u.l, {-1, -1}, 1e-12) && close(u.u, {1, 1}, 1e-12), b + ": a union of two rows");
}

/// Stated property: the bounds hold every point of random sets, for every option, and the exact
/// method is the tightest.
void bounds_property(const std::string &b) {
    cora::Rng rng(33);
    for (int k = 0; k < 4; ++k) {
        const int n = 3, m = 4, p = 3, B = 2;
        std::vector<double> c(B * n), G(B * n * m), A(B * p * m), rhs(B * p);
        rng.normal(c.data(), c.size(), 1.0);
        rng.normal(G.data(), G.size(), 1.0);
        rng.normal(A.data(), A.size(), 1.0);
        std::vector<double> extra(B * p);
        rng.normal(extra.data(), extra.size(), 1.0);
        // offsets that keep the origin inside, so no member is empty
        for (int i = 0; i < B * p; ++i) rhs[i] = 0.5 + std::abs(extra[i]);
        const Tensor ct = Tensor::from_data(c, {B, n, 1}), Gt = Tensor::from_data(G, {B, n, m}),
                     At = Tensor::from_data(A, {B, p, m}), bt = Tensor::from_data(rhs, {B, p, 1});
        Options batched, looped, exact;
        looped.batch_union = false;
        exact.method = "exact";
        const ConZonotope::Bounds fm =
            ConZonotope::approximate_bounds(ct, Gt, At, bt, std::nullopt, 1);
        const ConZonotope::Bounds loop =
            ConZonotope::approximate_bounds(ct, Gt, At, bt, std::nullopt, 1, looped);
        const ConZonotope::Bounds ex =
            ConZonotope::approximate_bounds(ct, Gt, At, bt, std::nullopt, 1, exact);
        check(close(fm.l, loop.l, 0) && close(fm.u, loop.u, 0),
              b + ": the loop over union rows agrees");
        check(fm.l.less_equal(ex.l + 1e-9).min().item() == 1.0 &&
                  fm.u.greater_equal(ex.u - 1e-9).min().item() == 1.0,
              b + ": the approximation holds the exact box");
        // the factor bounds agree
        check(fm.bl.less_equal(ex.bl + 1e-9).min().item() == 1.0,
              b + ": and the exact factor bounds");
    }
}

/// Wrong shapes, rows and methods are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Tensor c = Tensor::zeros({1, 2, 1}), G = Tensor::from_data({1, 0, 0, 1}, {1, 2, 2});
    const Tensor A = Tensor::from_data({1, 1}, {1, 1, 2}),
                 rhs = Tensor::from_data({-1.5}, {1, 1, 1});
    check(
        test::raises([&] { ConZonotope::approximate_bounds(c, G, Tensor::zeros({1, 1, 1}), rhs); },
                     Error::dimension_mismatch),
        "approximate_bounds: A has the columns of G");
    check(test::raises([&] { ConZonotope::approximate_bounds(c, G, A, Tensor::zeros({1, 2, 1})); },
                       Error::dimension_mismatch),
          "approximate_bounds: b has the rows of A");
    check(test::raises([&] { ConZonotope::approximate_bounds(c, G, A, rhs, std::nullopt, 2); },
                       Error::wrong_value),
          "approximate_bounds: more union rows than rows");
    Options bad;
    bad.method = "nothing";
    check(test::raises([&] { ConZonotope::approximate_bounds(c, G, A, rhs, std::nullopt, 1, bad); },
                       Error::wrong_field_value),
          "approximate_bounds: an unknown method");
    bad.method = "dual-iter";
    // an unported method
    check(test::raises([&] { ConZonotope::approximate_bounds(c, G, A, rhs, std::nullopt, 1, bad); },
                       Error::not_supported),
          "approximate_bounds: dual-iter is not ported");
    const Tensor on = ConZonotope::approximate_bounds(c, G, A, rhs).l;
    ChecksScope off(false);
    check(close(ConZonotope::approximate_bounds(c, G, A, rhs).l, on, 0),
          "approximate_bounds: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        bounds_known(b);
        bounds_property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope approximate_bounds");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
