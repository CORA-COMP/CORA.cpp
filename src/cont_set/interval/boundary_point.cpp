// boundary_point - the point where a ray leaves a box, as CORA's interval.boundaryPoint
//
// The bound of each coordinate that dir points to gives s. If the ray never leaves, +-Inf along
// dir (start where dir is 0). The bound is (sup - start) / dir for dir > 0 and (start - inf) /
// -dir for dir < 0, as CORALean evaluates it.
//
// Syntax:   Tensor x = I.boundary_point(dir);   Tensor x = I.boundary_point(dir, start);
// Inputs:   dir - direction (..., n), not zero
//           start - a point (..., n) of I, by default its center (required if that is NaN)
// Outputs:  x - start + s dir for the largest s with x in I, (..., n); NaN for an empty box
// See also: contains, center
//
// CORA: contSet/@interval/boundaryPoint.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The step s of the ray start + s dir to each coordinate's bound (Inf where dir is 0 or the
/// bound is infinite), (..., n).
Tensor aux_step_to_bound(const Interval &I, const Tensor &dir, const Tensor &start) {
    const Tensor zero = dir.zeros_like();
    const Tensor moves_up = dir.greater(zero), moves_down = dir.less(zero);
    const Tensor distance = Tensor::where(moves_up, I.sup() - start, start - I.inf());
    const Tensor speed = Tensor::where(moves_up, dir, -dir);
    // the divisor is 1 where dir is 0, so that no NaN reaches the masked entries
    const Tensor safe = Tensor::where(moves_up.add(moves_down), speed, zero + 1.0);
    const Tensor inf = dir.full_like(dir.shape(), std::numeric_limits<double>::infinity());
    return Tensor::where(moves_up.add(moves_down), distance.divide(safe), inf);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Interval::boundary_point(const Tensor &dir, const std::optional<Tensor> &start_point) const {
    detail::Entry entry;
    CORA_CHECK(elem_rank_ == 1 && is_vector(dir, dim()), Error::dimension_mismatch, [&] {
        return "boundary_point: dir must be a vector of the dimension " + std::to_string(dim()) +
               " of the interval, got the shape " + shape_text(dir.shape());
    });
    CORA_CHECK(dir.abs().max(-1, true).min().item() > machine_eps(dir.dtype()), Error::wrong_value,
               [] { return std::string("boundary_point: dir must be a nonzero vector"); });
    // the empty interval has no boundary: NaN
    if (is_empty()) return dir.full_like(dir.shape(), std::numeric_limits<double>::quiet_NaN());
    const Tensor start = start_point ? *start_point : center();
    CORA_CHECK(start.isnan().max().item() == 0, Error::wrong_value,
               [] { return std::string("boundary_point: an unbounded box needs a start point"); });
    CORA_CHECK(is_vector(start, dim()), Error::dimension_mismatch, [&] {
        return "boundary_point: the start point must be a vector of the dimension " +
               std::to_string(dim()) + " of the interval, got the shape " +
               shape_text(start.shape());
    });
    CORA_CHECK(contains(start), Error::wrong_value, [] {
        return std::string("boundary_point: the start point must be contained in the interval");
    });
    // the step to the nearest bound along dir
    const Tensor step = aux_step_to_bound(*this, dir, start).min(-1, true);
    const double kInf = std::numeric_limits<double>::infinity();
    // the ray never leaves where no bound is finite: it ends at +-Inf in the coordinates of dir
    const Tensor leaves = step.less(step.full_like(step.shape(), kInf));
    const Tensor at_infinity = Tensor::where(dir.equal(dir.zeros_like()), start, dir.sign() * kInf);
    return Tensor::where(leaves, dir.multiply(step) + start, at_infinity);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
