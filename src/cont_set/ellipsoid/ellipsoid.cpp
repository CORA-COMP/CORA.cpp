// ellipsoid - the constructors of an ellipsoid, checked, as CORA's ellipsoid
//
// Syntax:   Ellipsoid E(Q);   Ellipsoid E(Q, q);   Ellipsoid E(Q, q, tol);
// Inputs:   Q - shape matrix (..., n, n), finite, symmetric and positive semidefinite up to tol;
//           q - center (..., n, 1), finite, default 0
//           tol - bound of the asymmetry and of the
//           negative eigenvalues, kept as the tolerance of the object; default: the repository
//           tolerance (docs/architecture.md; CORA's is 1e-6). Q without entries needs q without
//           entries: the empty ellipsoid
// Outputs:  E - the ellipsoid {x | (x - q)' Q^+ (x - q) <= 1}
// See also: empty, Zonotope::Zonotope
//
// CORA: contSet/@ellipsoid/ellipsoid.m
// CORA authors: Victor Gassmann, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/ellipsoid/ellipsoid.h"
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

// Not ported yet ----------------------------------------------------------------------------------

// Not yet ported: the support function, random points, projections and vertices.
Tensor Ellipsoid::support_function_impl(const Tensor &) const {
    CORA_FAIL(Error::not_supported, "Ellipsoid: support_function is not ported yet");
}

Tensor Ellipsoid::rand_point_impl(int64_t, Rng &) const {
    CORA_FAIL(Error::not_supported, "Ellipsoid: rand_point is not ported yet");
}

Set Ellipsoid::project_impl(const std::vector<int64_t> &) const {
    CORA_FAIL(Error::not_supported, "Ellipsoid: project is not ported yet");
}

std::vector<Polygon> Ellipsoid::vertices_impl() const {
    CORA_FAIL(Error::not_supported, "Ellipsoid: vertices is not ported yet");
}

std::vector<std::vector<Point3>> Ellipsoid::vertices_3d_impl() const {
    CORA_FAIL(Error::not_supported, "Ellipsoid: vertices_3d is not ported yet");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
