// con_zonotope - the constructors of a constrained zonotope, checked, as CORA's conZonotope
//
// Syntax:   ConZonotope cz(c, G);   ConZonotope cz(c, G, A, b);   ConZonotope cz(Z);
//           ConZonotope cz = ConZonotope::from_matrix(Zfull, A, b);
// Inputs:   c - center (..., n, 1)
//           G - generators (..., n, m)
//           A - constraint matrix
//           (..., p, m);  b - constraint offset (..., p, 1); all finite, one batch shape
// Outputs:  cz - the set {c + G x | |x|_inf <= 1, A x = b}; no constraints give p = 0
// See also: empty, Zonotope::Zonotope
//
// CORA: contSet/@conZonotope/conZonotope.m
// CORA: contSet/@interval/conZonotope.m
// CORA: contSet/@zonotope/conZonotope.m
// CORA authors: Dmitry Grebenyuk, Mark Wetzlinger, Tobias Ladner, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cont_set/con_zonotope/private/priv.h"
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
    if (c.ndim() < 2 || c.shape().back() != 1)
        return fail("the center must be a column (..., n, 1), got the shape " +
                    shape_text(c.shape()));
    if (G.ndim() < 2 || G.shape()[G.ndim() - 2] != c.shape()[c.ndim() - 2])
        return fail("the dimensions of the center and the generator matrix do not match: " +
                    shape_text(c.shape()) + " and " + shape_text(G.shape()));
    if (A.ndim() < 2 || A.shape().back() != G.shape().back())
        return fail("the dimensions of the generator matrix and the constraint matrix do not "
                    "match: " +
                    shape_text(G.shape()) + " and " + shape_text(A.shape()));
    if (b.ndim() < 2 || b.shape().back() != 1)
        return fail("the constraint offset must be a column (..., p, 1), got the shape " +
                    shape_text(b.shape()));
    if (b.shape()[b.ndim() - 2] != A.shape()[A.ndim() - 2])
        return fail("the dimensions of the constraint matrix and the constraint offset do not "
                    "match: " +
                    shape_text(A.shape()) + " and " + shape_text(b.shape()));
    const auto batch = [](const Tensor &t) {
        // all four parts carry one batch shape
        return std::vector<int64_t>(t.shape().begin(), t.shape().end() - 2);
    };
    if (batch(c) != batch(G) || batch(c) != batch(A) || batch(c) != batch(b))
        return fail("the center, generators and constraints must have one batch shape");
    if (!aux_finite(c) || !aux_finite(G) || !aux_finite(A) || !aux_finite(b))
        return fail("the entries must be finite");
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope::ConZonotope(Tensor c, Tensor G, Tensor A, Tensor b)
    : ContSet(kClassId), c_(std::move(c)), G_(std::move(G)), A_(std::move(A)), b_(std::move(b)) {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(c_, G_, A_, b_);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
}

ConZonotope::ConZonotope(const Tensor &c, const Tensor &G)
    : ConZonotope(c, G, priv_zeros(G, 0, G.shape().back()), priv_zeros(c, 0, 1)) {}

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
    const Tensor c = Z.slice(-1, 0, 1), G = Z.slice(-1, 1, Z.shape().back());
    // the stored parts keep the batch shape of the center
    if (A) return {c, G, *A, *b};
    return {c, G};
}

// Not ported yet ----------------------------------------------------------------------------------

// Not yet ported: the support function, random points and vertices.
Tensor ConZonotope::support_function_impl(const Tensor &) const {
    CORA_FAIL(Error::not_supported, "ConZonotope: support_function is not ported yet");
}

Tensor ConZonotope::rand_point_impl(int64_t, Rng &) const {
    CORA_FAIL(Error::not_supported, "ConZonotope: rand_point is not ported yet");
}

std::vector<Polygon> ConZonotope::vertices_impl() const {
    CORA_FAIL(Error::not_supported, "ConZonotope: vertices is not ported yet");
}

std::vector<std::vector<Point3>> ConZonotope::vertices_3d_impl() const {
    CORA_FAIL(Error::not_supported, "ConZonotope: vertices_3d is not ported yet");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
