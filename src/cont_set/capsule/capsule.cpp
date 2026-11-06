// capsule - the constructors of a capsule, checked, as CORA's capsule
//
// All entries are finite, r is nonnegative, and they share one batch shape.
//
// Syntax:   Capsule C(c, g, r);   Capsule C(c, g);   Capsule C(c);
// Inputs:   c - center (..., n, 1)
//           g - generator (..., n, 1), default 0
//           r - radius (..., 1, 1) or a number, default 0
// Outputs:  C - the capsule {x | min_t |x - (c + t g)| <= r, t in [-1, 1]}
// See also: empty, origin, Zonotope::Zonotope
//
// CORA: contSet/@capsule/capsule.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cont_set/capsule/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether every entry of t is finite.
bool aux_finite(const Tensor &t) { return priv_all(t.detach().isfinite()); }

/// The shape of the radius that goes with a center of this shape: (..., 1, 1).
Shape aux_radius_shape(const Tensor &c) {
    std::vector<int64_t> shape = c.shape();
    if (shape.size() < 2) return Shape{1, 1};
    shape.back() = 1;
    shape[shape.size() - 2] = 1;
    return Shape(shape);
}

/// A tensor of the radius' shape like c, filled with value; none for an undefined c, which the
/// constructor refuses.
Tensor aux_radius_like(const Tensor &c, double value) {
    return c.defined() ? c.full_like(aux_radius_shape(c), value) : Tensor();
}

/// Why the data are no capsule, if they are none.
detail::Refusal aux_refusal(const Tensor &c, const Tensor &g, const Tensor &r) {
    const auto wrong_value = [](const std::string &text) {
        return detail::Refusal(std::pair{Error::wrong_value, "Capsule: " + text});
    };
    if (!c.defined() || !g.defined() || !r.defined())
        return wrong_value("the center, the generator and the radius must be tensors");
    // the center and the generator are columns of at least one row
    for (const Tensor *column : {&c, &g})
        if (column->ndim() < 2 || column->shape().back() != 1 ||
            column->shape()[column->ndim() - 2] < 1)
            return wrong_value("the center and the generator must be nonempty columns (..., n, 1), "
                               "got the shape " +
                               shape_text(column->shape()));
    if (c.shape() != g.shape())
        return detail::Refusal(
            std::pair{Error::wrong_input_in_constructor,
                      std::string("Capsule: Dimension of center and generator do "
                                  "not match.")});
    // one radius per capsule of the batch
    if (r.shape() != aux_radius_shape(c))
        return wrong_value("the radius must be a scalar (..., 1, 1) per capsule, got the shape " +
                           shape_text(r.shape()));
    if (!aux_finite(c) || !aux_finite(g) || !aux_finite(r))
        return wrong_value("the entries must be finite");
    if (!priv_all(r.detach().greater_equal(r.zeros_like())))
        return wrong_value("the radius must be nonnegative");
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Constructor -------------------------------------------------------------------------------------

Capsule::Capsule(Unchecked, Tensor c, Tensor g, Tensor r)
    : ContSet(kClassId), c_(std::move(c)), g_(std::move(g)), r_(std::move(r)) {}

Capsule::Capsule(const Tensor &c, Tensor g, Tensor r)
    : ContSet(kClassId), c_(c), g_(std::move(g)), r_(std::move(r)) {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(c_, g_, r_);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
}

Capsule::Capsule(const Tensor &c, Tensor g, double r)
    : Capsule(c, std::move(g), aux_radius_like(c, r)) {}

Capsule::Capsule(const Tensor &c, Tensor g) : Capsule(c, std::move(g), 0.0) {}

Capsule::Capsule(const Tensor &c) : Capsule(c, c.defined() ? c.zeros_like() : Tensor(), 0.0) {}

// Not ported yet ----------------------------------------------------------------------------------

// Not yet ported: the support function, random points, projections and vertices.
Tensor Capsule::support_function_impl(const Tensor &) const {
    CORA_FAIL(Error::not_supported, "Capsule: support_function is not ported yet");
}

Tensor Capsule::rand_point_impl(int64_t, Rng &) const {
    CORA_FAIL(Error::not_supported, "Capsule: rand_point is not ported yet");
}

Set Capsule::project_impl(const std::vector<int64_t> &) const {
    CORA_FAIL(Error::not_supported, "Capsule: project is not ported yet");
}

std::vector<Polygon> Capsule::vertices_impl() const {
    CORA_FAIL(Error::not_supported, "Capsule: vertices is not ported yet");
}

std::vector<std::vector<Point3>> Capsule::vertices_3d_impl() const {
    CORA_FAIL(Error::not_supported, "Capsule: vertices_3d is not ported yet");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
