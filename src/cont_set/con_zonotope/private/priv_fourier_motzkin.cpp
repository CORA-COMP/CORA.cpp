// priv_fourier_motzkin - the factor bounds of a batch of systems A x <= b in a box (ICLR Prop. 3)
//
// Each pass bounds every factor by every row from the least value of the row's other terms over
// the box the pass starts from (Jacobi), and intersects; the passes stop when no bound moves by
// more than `tighter` (a crossed bound, an empty member, also stops). Two orders of the sum of the
// other terms: "cora" takes the row's sum minus the term (CORA's aux_fourierMotzkinApproximation,
// O(p q) per pass); "coralean" sums the other terms in CORALean's order (Interval.contractHalfspace
// of ConZonotope.fourierMotzkinBounds, the rows in order, O(p q^2)), which on the lean backend is
// the oracle's bit for bit. A row with the offset +inf bounds nothing.
//
// Syntax:   auto [bl, bu] = priv_fourier_motzkin(A, b, bl0, bu0, passes, order, tighter);
// Inputs:   A (M, p, q), b (M, p) - the systems; bl0, bu0 (M, q) - the box to start from
// Outputs:  bl, bu (M, q) - the bounds of the factors
// See also: approximate_bounds.cpp, cora/nn/neural_network/verify.h
//
// CORA: contSet/@conZonotope/approximateBoundsWithGPU.m
// CORA authors: Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Extremes ----------------------------------------------------------------------------------------

/// CORALean's maximum and minimum of tensors: b where a <= b, else a (and a, else b).
Tensor aux_vmax(const Tensor &a, const Tensor &b) { return Tensor::where(a.less_equal(b), b, a); }
Tensor aux_vmin(const Tensor &a, const Tensor &b) { return Tensor::where(a.less_equal(b), a, b); }

// The order of CORA -------------------------------------------------------------------------------

/// The parts of A that every pass in CORA's order reads: the signs and A with ones for zeros.
struct CoraParts {
    Tensor positive, negative, safe;
};

CoraParts aux_cora_parts(const Tensor &A) {
    const Tensor zero = A.zeros_like();
    // a row without a coefficient bounds nothing: divide by one, the quotient is not used
    return {A.greater(zero), A.less(zero), Tensor::where(A.equal(zero), A.ones_like(), A)};
}

/// One pass in CORA's order: the bound of each factor by each row, the tightest over the rows.
std::pair<Tensor, Tensor> aux_pass_cora(const Tensor &A, const CoraParts &parts, const Tensor &b,
                                        const Tensor &bl, const Tensor &bu) {
    const Tensor low = bl.expand_dims(1), up = bu.expand_dims(1);
    // the least value of each term a_ij x_j (zero for a zero a_ij) and of the others in its row
    const Tensor terms = A.multiply(Tensor::where(parts.positive, low, up));
    const Tensor others = terms.sum(-1, true) - terms;
    const Tensor implied = (b.expand_dims(-1) - others).divide(parts.safe).maximum(low).minimum(up);
    // a negative coefficient gives a lower bound, a positive one an upper bound
    return {Tensor::where(parts.negative, implied, low).max(1),
            Tensor::where(parts.positive, implied, up).min(1)};
}

// The order of CORALean ---------------------------------------------------------------------------

/// One pass in CORALean's order: each row contracts the starting box (contractHalfspace), the
/// sum of the other terms nested to the right, and the result is intersected with the running box.
std::pair<Tensor, Tensor> aux_pass_coralean(const Tensor &A, const Tensor &b, const Tensor &bl,
                                            const Tensor &bu) {
    const int64_t p = A.shape()[1], q = A.shape()[2];
    const Tensor eye = A.eye_like(q).reshape({1, q, q});
    Tensor Jl = bl, Ju = bu;
    for (int64_t i = 0; i < p; ++i) {
        const Tensor a = A.slice(1, i, i + 1).squeeze(1), d = b.slice(1, i, i + 1);
        // s_j: the sum over k != j of max(-a_k sup_k, -a_k inf_k), a zero in place of term j
        const Tensor term = aux_vmax(a.negative().multiply(bu), a.negative().multiply(bl));
        const Tensor s = Tensor::where(eye, eye.zeros_like(), term.expand_dims(1)).sum(-1);
        const Tensor c = d + s, zero = a.zeros_like();
        const Tensor inf =
            Tensor::where(a.less(zero), aux_vmax(bl, c.divide(a.negative()).negative()), bl);
        const Tensor sup = Tensor::where(zero.less(a), aux_vmin(bu, c.divide(a)), bu);
        Jl = aux_vmax(Jl, inf);
        Ju = aux_vmin(Ju, sup);
    }
    return {Jl, Ju};
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<Tensor, Tensor> priv_fourier_motzkin(const Tensor &A, const Tensor &b, Tensor bl,
                                               Tensor bu, int64_t passes, FmOrder order,
                                               double tighter) {
    const CoraParts parts = order == FmOrder::cora ? aux_cora_parts(A) : CoraParts{};
    for (int64_t pass = 0; pass < passes; ++pass) {
        std::pair<Tensor, Tensor> next;
        switch (order) {
        case FmOrder::cora:
            next = aux_pass_cora(A, parts, b, bl, bu);
            break;
        case FmOrder::coralean:
            next = aux_pass_coralean(A, b, bl, bu);
            break;
        default:
            CORA_FAIL(Error::wrong_value, "fourier_motzkin: an unknown order");
        }
        // the passes end when no bound of a member with a point moves
        const Tensor lower_moved = (bl + tighter).less(next.first);
        const Tensor upper_moved = next.second.less(bu - tighter);
        const bool moved = lower_moved.maximum(upper_moved)
                               .multiply(next.first.less_equal(next.second))
                               .max()
                               .item() > 0;
        bl = next.first;
        bu = next.second;
        if (!moved) break;
    }
    return {bl, bu};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
