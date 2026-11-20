// con_zonotope - the constructors of a constrained zonotope, checked, as CORA's conZonotope
//
// All entries are finite and share one batch shape.
//
// Syntax:   ConZonotope cz(c, G);   ConZonotope cz(c, G, A, b);   ConZonotope cz(c, G, F);
//           ConZonotope cz(Z);   Polytope::LinearPrograms lp = cz.factor_programs();
//           ConZonotope cz = ConZonotope::from_matrix(Zfull, A, b);
// Inputs:   c - center (..., n)
//           G - generators (..., m, n), one per row
//           A - constraint matrix (..., p, m)
//           b - constraint offset (..., p)
//           F - factor polytope {x | A x = b}
// Outputs:  cz - the set {c + G x | |x|_inf <= 1, A x = b}; no constraints give p = 0
// See also: empty, Zonotope::Zonotope
//
// CORA: contSet/@conZonotope/conZonotope.m
// CORA: contSet/@interval/conZonotope.m
// CORA: contSet/@zonotope/conZonotope.m
// CORA authors: Dmitry Grebenyuk, Mark Wetzlinger, Tobias Ladner, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   20-November-2026
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cont_set/con_zonotope/private/priv.h"
#include "global/blocks.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Checks ------------------------------------------------------------------------------------------

/// Whether every entry of t is finite.
bool aux_finite(const Tensor &t) {
    if (t.numel() == 0) return true;
    return t.detach().isfinite().sum().item() == double(t.numel());
}

/// Why the data are no constrained zonotope, if they are none.
detail::Refusal aux_refusal(const Tensor &c, const Tensor &G, const Tensor &A, const Tensor &b) {
    const auto fail = [](const std::string &text) {
        return detail::Refusal(
            std::pair{Error::wrong_input_in_constructor, "ConZonotope: " + text});
    };
    if (!c.defined() || !G.defined() || !A.defined() || !b.defined())
        return fail("the center, the generators and the constraints must be tensors");
    if (c.ndim() < 1)
        return fail("the center must be a vector (..., n), got the shape " + shape_text(c.shape()));
    // one generator per row: G (..., m, n)
    if (G.ndim() < 2 || G.shape().back() != c.shape().back())
        return fail("the dimensions of the center and the generator rows do not match: " +
                    shape_text(c.shape()) + " and " + shape_text(G.shape()));
    if (A.ndim() < 2 || A.shape().back() != G.shape()[G.ndim() - 2])
        return fail("the dimensions of the generator matrix and the constraint matrix do not "
                    "match: " +
                    shape_text(G.shape()) + " and " + shape_text(A.shape()));
    if (b.ndim() < 1)
        return fail("the constraint offset must be a vector (..., p), got the shape " +
                    shape_text(b.shape()));
    // the constraint matrix and offset agree in rows
    if (b.shape().back() != A.shape()[A.ndim() - 2])
        return fail("the dimensions of the constraint matrix and the constraint offset do not "
                    "match: " +
                    shape_text(A.shape()) + " and " + shape_text(b.shape()));
    const auto batch = [](const Tensor &t, int64_t rank) {
        // all four parts carry one batch shape
        return std::vector<int64_t>(t.shape().begin(), t.shape().end() - rank);
    };
    if (batch(c, 1) != batch(G, 2) || batch(G, 2) != batch(A, 2) || batch(G, 2) != batch(b, 1))
        return fail("the center, generators and constraints must have one batch shape");
    if (!aux_finite(c) || !aux_finite(G) || !aux_finite(A) || !aux_finite(b))
        return fail("the entries must be finite");
    return std::nullopt;
}

/// Why the factor polytope does not fit the center and the generators, if it does not.
detail::Refusal aux_polytope_refusal(const Tensor &c, const Tensor &G, const Polytope &F) {
    const auto fail = [](const std::string &text) {
        return detail::Refusal(
            std::pair{Error::wrong_input_in_constructor, "ConZonotope: " + text});
    };
    if (!F.is_h_rep() || F.A().shape()[F.A().ndim() - 2] != 0)
        return fail("the factor polytope must be equalities A x = b only");
    return aux_refusal(c, G, F.Ae(), F.be());
}

} // namespace


// ===========================================  MAIN  =========================================== //

// From the data -----------------------------------------------------------------------------------

// the factor polytope {x | A x = b} of checked data: the equalities, no inequality rows
Polytope ConZonotope::factor_polytope_of(const Tensor &c, const Tensor &G, Tensor A, Tensor b) {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(c, G, A, b);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return Polytope::equalities(std::move(A), std::move(b));
}

ConZonotope::ConZonotope(Tensor c, Tensor G, Tensor A, Tensor b)
    : ContSet(kClassId), c_(std::move(c)), G_(std::move(G)),
      F_(factor_polytope_of(c_, G_, std::move(A), std::move(b))) {}

// the factor polytope as it is, checked against the center and the generators
ConZonotope::ConZonotope(Tensor c, Tensor G, Polytope F)
    : ContSet(kClassId), c_(std::move(c)), G_(std::move(G)),
      F_(std::move(F)) {
    detail::Entry entry;
    const detail::Refusal refusal = aux_polytope_refusal(c_, G_, F_);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
}

// the factors in the unit box
Polytope::LinearPrograms ConZonotope::factor_programs() const {
    const auto m = std::size_t(G_.shape()[G_.ndim() - 2]);
    return F_.linear_programs(std::vector<double>(m, -1.0), std::vector<double>(m, 1.0));
}

// From other sets --------------------------------------------------------------------------------

ConZonotope::ConZonotope(const Tensor &c, const Tensor &G)
    : ConZonotope(c, G, zeros_block(G, 0, G.shape()[G.ndim() - 2]),
                  zeros_block(G, 0, 1).squeeze(-1)) {}

ConZonotope::ConZonotope(const Zonotope &Z) : ConZonotope(Z.center(), Z.generators()) {}

ConZonotope::ConZonotope(const Interval &I) : ConZonotope(I.zonotope()) {}

ConZonotope ConZonotope::from_matrix(const Tensor &Z, const std::optional<Tensor> &A,
                                     const std::optional<Tensor> &b) {
    detail::Entry entry;
    CORA_CHECK(
        Z.defined() && Z.ndim() >= 2 && Z.shape().back() >= 1, Error::wrong_input_in_constructor,
        [] { return std::string("ConZonotope: Z = [c G] needs at least the center column"); });
    CORA_CHECK(A.has_value() == b.has_value(), Error::wrong_input_in_constructor, [] {
        return std::string("ConZonotope: give the constraint matrix and the offset together");
    });
    const Tensor c = Z.slice(-1, 0, 1).squeeze(-1), G = Z.slice(-1, 1, Z.shape().back());
    // the stored parts keep the batch shape of the center
    if (A) return {c, G.matrix_transpose(), *A, *b};
    return {c, G.matrix_transpose()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
