// intersect_strip - the intersection of a constrained zonotope with strips |C x - y| <= phi
//
// Syntax:   cZres = cz.intersect_strip(C, phi, y);   cZres = cz.intersect_strip(C, phi, y, method);
//           cZres = cz.intersect_strip(C, phi, y, lambda);   cZres = cz.intersect_strip(C, phi, y,
//           aux);
// Inputs:   C (k, n), phi (k, 1), y (k, 1) - the k strips
//           method - the gain lambda (n, k) is chosen by "normGen" (default), "alamo-FRad" (one
//                    strip) or "wang-FRad" (with the matrices E, F, A, C of an observer, aux), as
//                    in Zonotope::intersect_strip; "bravo" (one strip) returns the zonotope
//                    around the first strip without the constraints, as CORA; "svd" and "radius"
//                    (an optimizer) are not supported; a given lambda is used as it is
// Outputs:  cZres - exact for every lambda: center c + lambda (y - C c), generators
//           [(I - lambda C) G, lambda diag(phi)], the constraints of cz and one equality per strip
//           C G x - phi x_new = y - C c
// See also: intersect, Zonotope::intersect_strip
//
// CORA: contSet/@conZonotope/intersectStrip.m

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The set of the zonotope Z that the gain made (center, generators), with the constraints of cz
/// and the rows C G x - phi x_new = y - C c for the new factors.
ConZonotope aux_with_strips(const ConZonotope &cz, const Zonotope &Z, const Tensor &C,
                            const Tensor &phi, const Tensor &y) {
    const int64_t k = C.shape()[0];
    const Tensor CG = C.matmul(cz.generators());
    Shape widths = CG.shape();
    widths.back() = k;
    const Tensor rows = Tensor::concat({CG, -phi.diag().broadcast_to(widths)}, -1);
    const Tensor A = Tensor::concat(
        {Tensor::concat({cz.A(), priv_zeros(cz.A(), cz.A().shape()[cz.A().ndim() - 2], k)}, -1),
         rows},
        -2);
    return {Z.center(), Z.generators(), A, Tensor::concat({cz.b(), y - C.matmul(cz.c())}, -2)};
}

/// Why the method is refused for the strips, if it is: an unknown one, one that needs an optimizer
/// or an observer, one for a single strip, or 'bravo' for a batch.
detail::Refusal aux_method_refusal(const ConZonotope &cz, const Tensor &C,
                                   const std::string &method) {
    const bool known = method == "normGen" || method == "svd" || method == "radius" ||
                       method == "alamo-FRad" || method == "wang-FRad" || method == "bravo";
    if (!known) return std::pair{Error::wrong_value, "unknown method '" + method + "'"};
    if (method == "svd" || method == "radius")
        return std::pair{Error::not_supported,
                         "the method '" + method + "' needs an unconstrained optimizer"};
    if (method == "wang-FRad")
        return std::pair{Error::wrong_value,
                         std::string("the method 'wang-FRad' needs the matrices E, F, A, C of the "
                                     "observer: pass a StripObserver")};
    if (method == "alamo-FRad" && C.shape()[0] != 1)
        return std::pair{Error::special_error, "'" + method + "' is for a single strip"};
    if (method == "bravo" && cz.c().ndim() != 2)
        return std::pair{Error::not_supported,
                         std::string("'bravo' chooses among candidates per set; it supports a "
                                     "single set, not a batch")};
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                         const std::string &method) const {
    detail::Entry entry;
    Refusal refusal = priv_strip_refusal(dim(), C, phi, y);
    if (!refusal) refusal = aux_method_refusal(*this, C, method);
    CORA_CHECK(!refusal, refusal->first, [&] { return "intersect_strip: " + refusal->second; });
    const Zonotope Z = Zonotope(c_, G_).intersect_strip(C, phi, y, method);
    // CORA's 'bravo' returns the zonotope around the strip, without the constraints
    if (method == "bravo") return ConZonotope(Z);
    return aux_with_strips(*this, Z, C, phi, y);
}

ConZonotope ConZonotope::intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                         const Tensor &lambda) const {
    detail::Entry entry;
    // a given gain: any matrix of the right size gives an enclosure
    Refusal refusal = priv_strip_refusal(dim(), C, phi, y);
    if (!refusal) refusal = priv_gain_refusal(dim(), C, lambda);
    CORA_CHECK(!refusal, refusal->first, [&] { return "intersect_strip: " + refusal->second; });
    return aux_with_strips(*this, Zonotope(c_, G_).intersect_strip(C, phi, y, lambda), C, phi, y);
}

ConZonotope ConZonotope::intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                         const StripObserver &aux) const {
    detail::Entry entry;
    // the gain of "wang-FRad" from the observer's matrices
    Refusal refusal = priv_strip_refusal(dim(), C, phi, y);
    if (!refusal) refusal = priv_observer_refusal(dim(), aux);
    CORA_CHECK(!refusal, refusal->first, [&] { return "intersect_strip: " + refusal->second; });
    return aux_with_strips(*this, Zonotope(c_, G_).intersect_strip(C, phi, y, aux), C, phi, y);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
