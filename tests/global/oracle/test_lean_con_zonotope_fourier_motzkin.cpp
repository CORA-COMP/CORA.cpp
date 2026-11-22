// test_lean_con_zonotope_fourier_motzkin - Fourier-Motzkin in CORALean's order against the oracle
//
// ORACLE: contSet/@conZonotope/approximateBoundsWithGPU.m
//
// The factor bounds of {x in [-1, 1]^q | A x <= b} after k passes of Koller et al., ICLR 2026,
// Prop. 3 in CORALean's order (the sum of a row's other terms as Interval.contractHalfspace sums
// them, the rows in order) are conZonotope.fourierMotzkinBounds bit for bit, for random feasible
// systems of a batch and 1, 3 and 8 passes; the order of approximate_bounds is that option.
//
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/oracle/con_zonotope.h"
#include "oracle_testing.h"

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// CORALean's order of Fourier-Motzkin against the oracle, member by member (finite rows only).
void aux_check_fm(const std::string &what, const Tensor &A, const Tensor &b, int64_t passes) {
    const int64_t M = A.shape()[0], q = A.shape()[2];
    const Tensor one = A.ones_like({M, q});
    const auto [bl, bu] = priv_fourier_motzkin(A, b, -one, one, passes, FmOrder::coralean, 0.0);
    for (int64_t m = 0; m < M; ++m) {
        const Tensor Am = A.slice(0, m, m + 1).squeeze(0), bm = b.slice(0, m, m + 1).squeeze(0);
        const std::vector<int64_t> finite = bm.isfinite().nonzero();
        const lean::Interval I = lean::fourier_motzkin_bounds(
            Am.take(finite, 0), bm.take(finite, 0).expand_dims(-1), passes);
        oracle::check(what + " inf " + std::to_string(m), bl.slice(0, m, m + 1).reshape({q, 1}),
                      I.inf());
        oracle::check(what + " sup " + std::to_string(m), bu.slice(0, m, m + 1).reshape({q, 1}),
                      I.sup());
    }
}

/// A random feasible system of M members: rows A, offsets at a point inside the box plus slack.
std::pair<Tensor, Tensor> aux_system(oracle::Random &rnd, int64_t M, int64_t p, int64_t q) {
    std::vector<Tensor> As, bs;
    for (int64_t m = 0; m < M; ++m) {
        const Tensor A = rnd.matrix(p, q), x = rnd.point(q) * 0.25;
        As.push_back(A);
        bs.push_back(A.matmul(x.expand_dims(-1)).squeeze(-1) + 0.125);
    }
    return {Tensor::stack(As, 0), Tensor::stack(bs, 0)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    // random feasible systems of three members, the passes of verify and CORA's default
    oracle::run("conZonotope.fourierMotzkinBounds", [&](oracle::Random &rnd) {
        for (const auto &[p, q] :
             std::vector<std::pair<int64_t, int64_t>>{{1, 3}, {4, 5}, {7, 9}}) {
            const auto [A, b] = aux_system(rnd, 3, p, q);
            for (const int64_t passes : {1, 3, 8}) aux_check_fm("fm", A, b, passes);
        }
    });

    // approximate_bounds with the order: the same factor bounds (its passes stop earlier)
    oracle::run("conZonotope.fourierMotzkinBounds", [&](oracle::Random &rnd) {
        const auto [A, b] = aux_system(rnd, 2, 4, 5);
        ConZonotope::BoundsOptions options;
        options.order = "coralean";
        const Tensor c = A.zeros_like({2, 3}),
                     G = rnd.matrix(5, 3).expand_dims(0).broadcast_to({2, 5, 3});
        const ConZonotope::Bounds r =
            ConZonotope::approximate_bounds(c, G, A, b, std::nullopt, 1, options);
        const Tensor one = A.ones_like({2, 5});
        const auto [bl, bu] = priv_fourier_motzkin(A, b, -one, one, options.max_iter,
                                                   FmOrder::coralean, numerics::kBoundTighter);
        oracle::check("approximate_bounds bl", r.bl, bl);
        oracle::check("approximate_bounds bu", r.bu, bu);
    });
    return test::finish("Fourier-Motzkin against the oracle");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
