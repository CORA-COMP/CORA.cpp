// ellipsoid - the constructors of an ellipsoid, checked, as CORA's ellipsoid
//
// tol is kept as the tolerance of the object (CORA's is 1e-6). Q without entries needs q without
// entries: the empty ellipsoid.
//
// Also the ellipsoids of a box (radii widened to 5e-3, n times its generators' ellipsoid) and of a
// zonotope of n independent generators (the SDP-based modes are not supported).
//
// Syntax:   Ellipsoid E(Q);   Ellipsoid E(Q, q);   Ellipsoid E(Q, q, tol);   Ellipsoid E(I, mode);
//           Ellipsoid E(Z, mode);
// Inputs:   Q - shape matrix (..., n, n), finite, symmetric and positive semidefinite up to tol
//           q - center (..., n, 1), finite, default 0
//           tol - bound of the asymmetry and the negative eigenvalues; default: the repository one
//           I - box
//           Z - zonotope (n, m), not a batch
//           mode - "outer" or "inner" for a box, one of the zonotope modes for a zonotope
// Outputs:  E - the ellipsoid {x | (x - q)' Q^+ (x - q) <= 1}
// See also: empty, Zonotope::Zonotope
//
// CORA: contSet/@ellipsoid/ellipsoid.m
// CORA: contSet/@interval/ellipsoid.m
// CORA: contSet/@zonotope/ellipsoid.m
// CORA authors: Victor Gassmann, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The zero center (..., n, 1) of a shape matrix (..., n, n); Q itself if it is no matrix.
Tensor aux_default_center(const Tensor &Q) {
    if (!Q.defined() || Q.ndim() < 2) return Q;
    Shape shape = Q.shape();
    shape.back() = 1;
    return Q.zeros_like(shape);
}

/// Whether every entry of t is finite.
bool aux_finite(const Tensor &t) {
    if (t.numel() == 0) return true;
    return t.detach().isfinite().sum().item() == double(t.numel());
}

/// The batch axes of a matrix or column.
std::vector<int64_t> aux_batch(const Tensor &t) { return {t.shape().begin(), t.shape().end() - 2}; }

/// Why the data are no ellipsoid, if they are none (CORA's checks of the constructor).
detail::Refusal aux_refusal(const Tensor &Q, const Tensor &q, double tol) {
    const auto fail = [](const std::string &text) {
        return detail::Refusal(std::pair{Error::wrong_input_in_constructor, "Ellipsoid: " + text});
    };
    if (Q.ndim() < 2 || q.ndim() < 2) return fail("the shape matrix must be a matrix (..., n, n)");
    const int64_t rows = Q.shape()[Q.ndim() - 2], cols = Q.shape().back();
    if (Q.numel() == 0 && rows == 0 && cols == 0) {
        if (q.numel() != 0) return fail("Shape matrix is empty, but center is not.");
        return std::nullopt;
    }
    // the remaining checks are those of CORA's inputArgsCheck
    if (!aux_finite(Q) || !aux_finite(q)) return fail("the entries must be finite");
    if (rows != cols) return fail("The shape matrix needs to be a square matrix.");
    if (q.shape().back() != 1)
        return fail("the center must be a column (..., n, 1), got the shape " +
                    shape_text(q.shape()));
    if (q.shape()[q.ndim() - 2] != rows)
        return fail("Dimensions of the shape matrix and center are different.");
    if (aux_batch(Q) != aux_batch(q))
        return fail("the shape matrix and the center must have one "
                    "batch shape");
    // symmetric up to tol and no eigenvalue below -tol
    const bool symmetric = within_tol(Q, Q.matrix_transpose(), tol).min().item() != 0.0;
    if (!symmetric || Q.eigh().first.min().item() < -tol)
        return fail("The shape matrix needs to be positive semidefinite/symmetric.");
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Constructor -------------------------------------------------------------------------------------

Ellipsoid::Ellipsoid(Tensor Q, Tensor q, Tol tol)
    : ContSet(kClassId), Q_(std::move(Q)), q_(std::move(q)), tol_(0.0) {
    detail::Entry entry;
    CORA_CHECK(Q_.defined() && q_.defined(), Error::wrong_input_in_constructor, [] {
        return std::string("Ellipsoid: the shape matrix and the center must be tensors");
    });
    tol_ = resolve_tol(entry, tol, Q_.dtype(), "Ellipsoid");
    const detail::Refusal refusal = aux_refusal(Q_, q_, tol_);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
}

Ellipsoid::Ellipsoid(const Tensor &Q, Tol tol) : Ellipsoid(Q, aux_default_center(Q), tol) {}

// Conversions -------------------------------------------------------------------------------------

Ellipsoid::Ellipsoid(const Interval &I, const std::string &mode) : Ellipsoid(Ellipsoid::empty(0)) {
    detail::Entry entry;
    // only the mode "outer" is available, and only for a nonempty box
    CORA_CHECK(mode == "outer" || mode == "inner", Error::wrong_value, [&] {
        return "Ellipsoid: the mode must be 'outer' or 'inner', got '" + mode + "'";
    });
    CORA_CHECK(mode == "outer", Error::not_supported,
               [] { return std::string("Ellipsoid: the mode 'inner' needs an SDP solver"); });
    CORA_CHECK(!I.is_empty(), Error::empty_set,
               [] { return std::string("Ellipsoid: the box is empty"); });
    // radii below the minimum are widened to it, unless they are 0
    const Tensor rad = I.rad();
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
    *this = Ellipsoid(widened.square().diag() * n, I.center());
}

Ellipsoid::Ellipsoid(const Zonotope &Z, const std::string &mode) : Ellipsoid(Ellipsoid::empty(0)) {
    detail::Entry entry;
    CORA_CHECK(mode == "outer:exact" || mode == "outer:norm" || mode == "outer:norm_bnd" ||
                   mode == "inner:exact" || mode == "inner:norm",
               Error::wrong_value, [&] { return "Ellipsoid: unknown mode '" + mode + "'"; });
    CORA_CHECK(Z.center().ndim() == 2, Error::not_supported, [] {
        return std::string("Ellipsoid: a zonotope converts one at a time, not as a batch");
    });
    const Tensor &c = Z.center(), &G = Z.generators();
    const int64_t n = Z.dim();
    // a point: no generator, or generators of length 0
    if (G.shape().back() == 0 || G.abs().max().item() == 0.0) {
        *this = Ellipsoid(c.zeros_like({n, n}), c);
        return;
    }
    CORA_CHECK(G.shape().back() == n && G.matrix_rank() == n, Error::not_supported, [] {
        return std::string(
            "Ellipsoid: only a zonotope of n independent generators converts without an SDP");
    });
    // the ellipsoid through the corners is n times larger than the inscribed one
    const double factor = mode.rfind("outer", 0) == 0 ? static_cast<double>(n) : 1.0;
    *this = Ellipsoid(G.matmul(G.matrix_transpose()) * factor, c);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
