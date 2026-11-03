// intersect_strip - a zonotope that encloses the intersection with strips |C x - y| <= phi
//
// Syntax:   Zres = Z.intersect_strip(C, phi, y);   Zres = Z.intersect_strip(C, phi, y, method);
//           Zres = Z.intersect_strip(C, phi, y, lambda);   Zres = Z.intersect_strip(C, phi, y,
//           aux);
// Inputs:   C (k, n), phi (k, 1), y (k, 1) - the k strips
//           method - the gain lambda (n, k) is chosen by "normGen" (default: the generators
//                    against the strip widths), "alamo-FRad" or "bravo" (the first strip only),
//                    or "wang-FRad" with the matrices E, F, A, C of an observer (aux);
//                    (CORA's "wang-FRad" gain, built from G G' of the zonotope itself, is not
//                    the optimal gain of Wang et al. 2018; every gain gives an enclosure);
//                    "svd", "radius" (an optimizer) and "alamo-volume" (a search over the
//                    generator subsets) are not supported; a given lambda is used as it is
// Outputs:  Zres - center c + lambda (y - C c), generators [(I - lambda C) G, lambda diag(phi)];
//           it contains the intersection for every lambda
// See also: intersect, enclose
//
// CORA: contSet/@zonotope/intersectStrip.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Gains -------------------------------------------------------------------------------------------

/// The zonotope for the gain lambda (n, k): CORA's aux_zonotopeFromLambda.
Zonotope aux_from_lambda(const Zonotope &Z, const Tensor &phi, const Tensor &C, const Tensor &y,
                         const Tensor &lambda) {
    const Tensor &c = Z.center();
    const Tensor c_new = c + lambda.matmul(y - C.matmul(c));
    const Tensor I = c.eye_like(Z.dim());
    const Tensor G_new = Tensor::concat(
        {(I - lambda.matmul(C)).matmul(Z.generators()), lambda.matmul(phi.diag())}, -1);
    return {c_new, G_new};
}

/// The gain "normGen": G G' C' (C G G' C' + diag(phi^2))^-1.
Tensor aux_gain_norm_gen(const Zonotope &Z, const Tensor &C, const Tensor &phi) {
    const Tensor &G = Z.generators();
    const Tensor GG = G.matmul(G.matrix_transpose());
    const Tensor numerator = GG.matmul(C.matrix_transpose());
    const Tensor den = C.matmul(numerator) + phi.square().diag();
    return numerator.matmul(den.inv());
}

/// The gain "alamo-FRad" for one strip: G G' c1' / (c1 G G' c1' + phi1^2).
Tensor aux_gain_alamo_frad(const Zonotope &Z, const Tensor &C, const Tensor &phi) {
    const Tensor &G = Z.generators();
    const Tensor GG = G.matmul(G.matrix_transpose());
    const Tensor c1 = C.slice(-2, 0, 1);
    const Tensor numerator = GG.matmul(c1.matrix_transpose());
    return numerator.divide(c1.matmul(numerator) + phi.slice(-2, 0, 1).square());
}

/// The gain "wang-FRad": R C' (C R C' + F F')^-1 with R = A G G' A' + E E'.
Tensor aux_gain_wang_frad(const Zonotope &Z, const Zonotope::StripObserver &aux) {
    const Tensor &G = Z.generators();
    const Tensor P = G.matmul(G.matrix_transpose());
    const Tensor Rbar =
        aux.A.matmul(P).matmul(aux.A.matrix_transpose()) + aux.E.matmul(aux.E.matrix_transpose());
    const Tensor S = aux.C.matmul(Rbar).matmul(aux.C.matrix_transpose()) +
                     aux.F.matmul(aux.F.matrix_transpose());
    return Rbar.matmul(aux.C.matrix_transpose()).matmul(S.inv());
}

// Bravo -------------------------------------------------------------------------------------------

/// The zonotope of "bravo" for the first strip: of the zonotope itself and one candidate per
/// generator that crosses the strip, the one of least det(G G'), the first on a tie.
Zonotope aux_bravo(const Zonotope &Z, const Tensor &C, const Tensor &phi, const Tensor &y) {
    const Tensor &G = Z.generators(), &p = Z.center();
    const int64_t n = Z.dim(), m = G.shape().back();
    const Tensor c1 = C.slice(-2, 0, 1), phi1 = phi.slice(-2, 0, 1), y1 = y.slice(-2, 0, 1);
    const Tensor along = c1.matmul(G); // (1, m): the strip's row against each generator
    const double limit = 1e10 * std::numeric_limits<double>::epsilon();
    const std::vector<double> a = along.data();
    // det(G G') is the product of the squared singular values, and 0 for fewer generators than rows
    const auto volume = [&](const Tensor &Gk) {
        return m < n ? 0.0 : Gk.svd()[1].square().prod(-2).item();
    };
    Zonotope best = Z;
    double least = volume(G);
    for (int64_t j = 0; j < m; ++j) {
        if (!(std::abs(a[j]) > limit)) continue; // the candidate is Z itself, no better than it
        const Tensor gj = G.slice(-1, j, j + 1), aj = along.slice(-1, j, j + 1);
        // every generator minus its share along g_j, and g_j scaled to the width of the strip
        const Tensor shifted = G - gj.matmul(along.divide(aj));
        const Tensor at_j = G.arange_like(m).equal(G.full_like({1}, double(j))).reshape({1, m});
        const Tensor T =
            Tensor::where(at_j, gj.multiply(phi1.divide(aj)).broadcast_to(G.shape()), shifted);
        const Tensor v = p + gj.multiply((y1 - c1.matmul(p)).divide(aj));
        const double vol = volume(T);
        if (vol < least) {
            least = vol;
            best = Zonotope(v, T);
        }
    }
    return best;
}

// Checks ------------------------------------------------------------------------------------------

/// Why the strips are refused, if they are: C (k, n), phi and y (k, 1).
Refusal aux_refusal(const Zonotope &Z, const Tensor &C, const Tensor &phi, const Tensor &y) {
    if (C.ndim() != 2 || C.shape()[1] != Z.dim())
        return std::pair{Error::dimension_mismatch,
                         "C must be a matrix (k, " + std::to_string(Z.dim()) + "), got the shape " +
                             shape_text(C.shape())};
    const int64_t k = C.shape()[0];
    if (phi.ndim() != 2 || !is_column(phi, k))
        return std::pair{Error::dimension_mismatch,
                         "phi must be a column (" + std::to_string(k) +
                             ", 1), one width per strip, got the shape " + shape_text(phi.shape())};
    if (y.ndim() != 2 || !is_column(y, k))
        return std::pair{Error::dimension_mismatch,
                         "y must be a column (" + std::to_string(k) +
                             ", 1), one center per strip, got the shape " + shape_text(y.shape())};
    return std::nullopt;
}

/// Why the gain is refused, if it is: one row per dimension and one column per strip.
Refusal aux_gain_refusal(const Zonotope &Z, const Tensor &C, const Tensor &lambda) {
    if (lambda.ndim() == 2 && lambda.shape()[0] == Z.dim() && lambda.shape()[1] == C.shape()[0])
        return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "the gain must be a matrix (" + std::to_string(Z.dim()) + ", " +
                         std::to_string(C.shape()[0]) + "), got the shape " +
                         shape_text(lambda.shape())};
}

/// Why the observer is refused, if it is: A (n, n), E (n, .), C (p, n) and F (p, .).
Refusal aux_observer_refusal(int64_t n, const Zonotope::StripObserver &aux) {
    const auto rows = [](const Tensor &T) { return T.ndim() == 2 ? T.shape()[0] : -1; };
    const auto cols = [](const Tensor &T) { return T.ndim() == 2 ? T.shape()[1] : -1; };
    if (rows(aux.A) == n && cols(aux.A) == n && rows(aux.E) == n && cols(aux.C) == n &&
        rows(aux.F) == rows(aux.C) && rows(aux.F) >= 0)
        return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     std::string("the observer needs A (n, n), E (n, .), C (p, n) and F (p, .)")};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                   const std::string &method) const {
    detail::Entry entry;
    const Refusal refusal = aux_refusal(*this, C, phi, y);
    CORA_CHECK(!refusal, refusal->first, [&] { return "intersect_strip: " + refusal->second; });
    // the methods CORA has, and which of them are supported
    const bool known = method == "normGen" || method == "svd" || method == "radius" ||
                       method == "alamo-FRad" || method == "alamo-volume" ||
                       method == "wang-FRad" || method == "bravo";
    CORA_CHECK(known, Error::wrong_value,
               [&] { return "intersect_strip: unknown method '" + method + "'"; });
    CORA_CHECK(method != "svd" && method != "radius" && method != "alamo-volume",
               Error::not_supported, [&] {
                   return "intersect_strip: the method '" + method + "' needs an " +
                          (method == "alamo-volume" ? "enumeration of the generator subsets"
                                                    : "unconstrained optimizer");
               });
    CORA_CHECK(method != "wang-FRad", Error::wrong_value, [] {
        return std::string("intersect_strip: the method 'wang-FRad' needs the matrices E, F, A, "
                           "C of the observer: pass a StripObserver");
    });
    CORA_CHECK(method == "normGen" || method == "bravo" || C.shape()[0] == 1, Error::special_error,
               [&] { return "intersect_strip: '" + method + "' is for a single strip"; });
    CORA_CHECK(method != "bravo" || c_.ndim() == 2, Error::not_supported, [] {
        return std::string("intersect_strip: 'bravo' chooses among candidates per zonotope; it "
                           "supports a single zonotope, not a batch");
    });
    // "bravo" builds the zonotope itself; the others differ in the gain only
    if (method == "bravo") {
        if (C.shape()[0] > 1)
            warn("CORA:contSet", "intersect_strip: 'bravo' only uses the first strip");
        return aux_bravo(*this, C, phi, y);
    }
    return aux_from_lambda(*this, phi, C, y,
                           method == "normGen" ? aux_gain_norm_gen(*this, C, phi)
                                               : aux_gain_alamo_frad(*this, C, phi));
}

Zonotope Zonotope::intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                   const Tensor &lambda) const {
    detail::Entry entry;
    // a given gain: any matrix of the right size gives an enclosure
    Refusal refusal = aux_refusal(*this, C, phi, y);
    if (!refusal) refusal = aux_gain_refusal(*this, C, lambda);
    CORA_CHECK(!refusal, refusal->first, [&] { return "intersect_strip: " + refusal->second; });
    return aux_from_lambda(*this, phi, C, y, lambda);
}

Zonotope Zonotope::intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                   const StripObserver &aux) const {
    detail::Entry entry;
    // the gain of "wang-FRad" from the observer's matrices
    Refusal refusal = aux_refusal(*this, C, phi, y);
    if (!refusal) refusal = aux_observer_refusal(dim(), aux);
    CORA_CHECK(!refusal, refusal->first, [&] { return "intersect_strip: " + refusal->second; });
    const Tensor lambda = aux_gain_wang_frad(*this, aux);
    const Refusal gain = aux_gain_refusal(*this, C, lambda);
    CORA_CHECK(!gain, gain->first, [&] { return "intersect_strip: " + gain->second; });
    return aux_from_lambda(*this, phi, C, y, lambda);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
