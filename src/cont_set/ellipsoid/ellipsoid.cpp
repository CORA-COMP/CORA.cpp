// ellipsoid - the constructors of an ellipsoid, checked, as CORA's ellipsoid
//
// tol is kept as the tolerance of the object (CORA's is 1e-6). The center comes first, as in every
// set (#163); the empty ellipsoid is Ellipsoid::empty(n), flagged in the mask.
//
// Also the ellipsoids of a box (radii widened to 5e-3, n times its generators' ellipsoid) and of a
// zonotope of n independent generators (the SDP-based modes are not supported).
//
// Syntax:   Ellipsoid E(Q);   Ellipsoid E(q, Q);   Ellipsoid E(q, Q, tol);   Ellipsoid E(I,
// method);
//           Ellipsoid E(Z, method);
// Inputs:   q - center (..., n), finite, default 0
//           Q - shape matrix (..., n, n), finite, symmetric and positive semidefinite up to tol
//           tol - bound of the asymmetry and the negative eigenvalues; default: the repository one
//           I - box
//           Z - zonotope (n) with generators (m, n), not a batch
//           method - "outer" or "inner" for a box, one of the zonotope modes for a zonotope
// Outputs:  E - the ellipsoid {x | (x - q)' Q^+ (x - q) <= 1}
// See also: empty, Zonotope::Zonotope
// References:
//    [1] A. A. Kurzhanskiy, P. Varaiya. "Ellipsoidal toolbox (ET)", CDC 2006, pp. 1498-1503.
//    [2] V. Gassmann, M. Althoff. "Scalable zonotope-ellipsoid conversions using the Euclidean
//        zonotope norm", ACC 2020.
//
// CORA: contSet/@ellipsoid/ellipsoid.m
// CORA: contSet/@interval/ellipsoid.m
// CORA: contSet/@zonotope/ellipsoid.m
// CORA authors: Victor Gassmann, Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The zero center (..., n) of a shape matrix (..., n, n); Q itself if it is no matrix.
Tensor aux_default_center(const Tensor &Q) {
    if (!Q.defined() || Q.ndim() < 2) return Q;
    const Shape shape(std::vector<int64_t>(Q.shape().begin(), Q.shape().end() - 1));
    return Q.zeros_like(shape);
}

/// Whether every entry of t is finite.
bool aux_finite(const Tensor &t) {
    if (t.numel() == 0) return true;
    return t.detach().isfinite().sum().item() == double(t.numel());
}

/// Why the data are no ellipsoid, if they are none (CORA's checks of the constructor).
detail::Refusal aux_refusal(const Tensor &q, const Tensor &Q, double tol) {
    const auto fail = [](const std::string &text) {
        return detail::Refusal(std::pair{Error::wrong_input_in_constructor, "Ellipsoid: " + text});
    };
    if (Q.ndim() < 2) return fail("the shape matrix must be a matrix (..., n, n)");
    // the center is a vector (..., n): one axis fewer than Q, so swapped arguments fail here
    if (q.ndim() != Q.ndim() - 1)
        return fail("the center must be a vector (..., n) and the shape matrix (..., n, n), "
                    "the center first; got the shapes " +
                    shape_text(q.shape()) + " and " + shape_text(Q.shape()));
    const int64_t rows = Q.size(-2), cols = Q.size(-1);
    // the remaining checks are those of CORA's inputArgsCheck
    if (!aux_finite(Q) || !aux_finite(q)) return fail("the entries must be finite");
    if (rows != cols) return fail("The shape matrix needs to be a square matrix.");
    if (q.size(-1) != rows) return fail("Dimensions of the shape matrix and center are different.");
    const std::vector<int64_t> q_batch(q.shape().begin(), q.shape().end() - 1),
        Q_batch(Q.shape().begin(), Q.shape().end() - 2);
    if (q_batch != Q_batch)
        return fail("the shape matrix and the center must have one batch shape");
    if (Q.numel() == 0) return std::nullopt;
    // symmetric up to tol and no eigenvalue below -tol
    const bool symmetric = within_tol(Q, Q.matrix_transpose(), tol).min().item() != 0.0;
    if (!symmetric || Q.eigh().first.min().item() < -tol)
        return fail("The shape matrix needs to be positive semidefinite/symmetric.");
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Constructor -------------------------------------------------------------------------------------

Ellipsoid::Ellipsoid(Tensor q, Tensor Q, Tol tol)
    : ContSet(kClassId), Q_(std::move(Q)), q_(std::move(q)), tol_(0.0) {
    detail::Entry entry;
    CORA_CHECK(Q_.defined() && q_.defined(), Error::wrong_input_in_constructor, [] {
        return std::string("Ellipsoid: the shape matrix and the center must be tensors");
    });
    tol_ = resolve_tol(entry, tol, Q_.dtype(), "Ellipsoid");
    const detail::Refusal refusal = aux_refusal(q_, Q_, tol_);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
}

Ellipsoid::Ellipsoid(const Tensor &Q, Tol tol) : Ellipsoid(aux_default_center(Q), Q, tol) {}

// Conversions -------------------------------------------------------------------------------------

Ellipsoid::Ellipsoid(const Interval &I, const std::string &method)
    : Ellipsoid(Ellipsoid::empty(0)) {
    detail::Entry entry;
    // only the method "outer" is available, and only for a nonempty box
    CORA_CHECK(method == "outer" || method == "inner", Error::wrong_value, [&] {
        return "Ellipsoid: the method must be 'outer' or 'inner', got '" + method + "'";
    });
    CORA_CHECK(method == "outer", Error::not_supported,
               [] { return std::string("Ellipsoid: the method 'inner' needs an SDP solver"); });
    CORA_CHECK(!I.is_empty(), Error::empty_set,
               [] { return std::string("Ellipsoid: the box is empty"); });
    // radii below the minimum are widened to it, unless they are 0
    const Tensor rad = I.half_width();
    const Tensor positive = rad.greater(rad.zeros_like());
    const Tensor widened = Tensor::where(
        rad.less(rad.full_like(rad.shape(), numerics::kBoxRadiusMin)),
        Tensor::where(positive, rad.full_like(rad.shape(), numerics::kBoxRadiusMin), rad), rad);
    // a flat edge leaves fewer than n generators: the box needs the minimum volume ellipsoid
    CORA_CHECK(widened.min().item() > 0.0 || widened.max().item() == 0.0, Error::not_supported, [] {
        return std::string(
            "Ellipsoid: a box with a flat edge needs the minimum volume enclosing ellipsoid");
    });
    const double n = static_cast<double>(I.dim());
    *this = Ellipsoid(I.center(), widened.expand_dims(-1).square().diag() * n);
}

Ellipsoid::Ellipsoid(const Zonotope &Z, const std::string &method)
    : Ellipsoid(Ellipsoid::empty(0)) {
    detail::Entry entry;
    CORA_CHECK(method == "outer:exact" || method == "outer:norm" || method == "outer:norm_bnd" ||
                   method == "inner:exact" || method == "inner:norm",
               Error::wrong_value, [&] { return "Ellipsoid: unknown method '" + method + "'"; });
    CORA_CHECK(Z.center().ndim() == 1, Error::not_supported, [] {
        return std::string("Ellipsoid: a zonotope converts one at a time, not as a batch");
    });
    // the generators are the rows of G (m, n): G' G is CORA's G G' of the columns
    const Tensor &c = Z.center(), &G = Z.generators();
    const int64_t n = Z.dim();
    // a point: no generator, or generators of length 0
    if (G.size(-2) == 0 || G.abs().max().item() == 0.0) {
        *this = Ellipsoid(c, c.zeros_like({n, n}));
        return;
    }
    CORA_CHECK(G.size(-2) == n && G.matrix_rank() == n, Error::not_supported, [] {
        return std::string(
            "Ellipsoid: only a zonotope of n independent generators converts without an SDP");
    });
    // the ellipsoid through the corners is n times larger than the inscribed one
    const double factor = method.rfind("outer", 0) == 0 ? static_cast<double>(n) : 1.0;
    *this = Ellipsoid(c, G.matrix_transpose().matmul(G) * factor);
}

// Conversions from the other set classes ----------------------------------------------------------

Ellipsoid::Ellipsoid(const Polytope &P, const std::string &method)
    : Ellipsoid(P.ellipsoid(method)) {}

Ellipsoid::Ellipsoid(const Capsule &C, const std::string &method)
    : Ellipsoid(C.ellipsoid(method)) {}

Ellipsoid::Ellipsoid(const PolyZonotope &)
    : Ellipsoid(detail::refuse_conversion<Ellipsoid>(
          "Ellipsoid(PolyZonotope): CORA bounds the radius by a global optimization of the "
          "support function (contSet/@polyZonotope/ellipsoid.m)")) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
