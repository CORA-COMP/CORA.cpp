// zonotope_norm - the norm of a point with the zonotope (centered at the origin) as unit ball
//
// Syntax:   s = Z.zonotope_norm(p);   [s, beta] = Z.zonotope_norm_minimizer(p);
// Inputs:   p - a point (n, 1)
// Outputs:  s - (1, 1): min s with G beta = p and |beta|_inf <= s, by a linear program (the
//           center of Z is ignored), infinity if p is outside the span of the generators;
//           beta - the minimizer (m, 1), an empty (0, 1) if s is infinite; with a center other
//           than the origin it is not a valid minimizer and a warning says so. Not a batch
// See also: contains, minnorm
//
// CORA: contSet/@zonotope/zonotopeNorm.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the call is refused, if it is: a batch, or a point that is not a column of the dimension.
Refusal aux_refusal(const Zonotope &Z, const Tensor &p) {
    if (Z.center().ndim() != 2)
        return std::pair{Error::not_supported,
                         std::string("a linear program per zonotope: it supports a single "
                                     "zonotope, not a batch")};
    if (p.shape() != Shape{Z.dim(), 1})
        return std::pair{Error::dimension_mismatch,
                         "the point must be a column (" + std::to_string(Z.dim()) + ", 1), got "
                             "the shape " + shape_text(p.shape())};
    return std::nullopt;
}

/// The norm and the minimizer on the host.
ZonotopeNorm aux_solve(const Zonotope &Z, const Tensor &p) {
    return priv_zonotope_norm_minimizer(priv_host_zonotope(Z), p.data());
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::zonotope_norm(const Tensor &p) const {
    detail::Entry entry;
    const Refusal refusal = aux_refusal(*this, p);
    CORA_CHECK(!refusal, refusal->first, [&] { return "zonotope_norm: " + refusal->second; });
    return Tensor::like(G_, {aux_solve(*this, p).value}, {1, 1});
}

std::pair<Tensor, Tensor> Zonotope::zonotope_norm_minimizer(const Tensor &p) const {
    detail::Entry entry;
    const Refusal refusal = aux_refusal(*this, p);
    CORA_CHECK(!refusal, refusal->first,
               [&] { return "zonotope_norm_minimizer: " + refusal->second; });
    const ZonotopeNorm solution = aux_solve(*this, p);
    const double tol = resolve_tol(Tol{}, c_.dtype());
    if (within_tol(c_, c_.zeros_like(), tol).all(-2).all(-1).item() == 0.0)
        warn("CORA:contSet", "zonotope_norm_minimizer: the minimizer might not be valid as the "
                             "zonotope is not zero-centered");
    const int64_t m = int64_t(solution.minimizer.size());
    return {Tensor::like(G_, {solution.value}, {1, 1}),
            Tensor::like(G_, solution.minimizer, {m, 1})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
