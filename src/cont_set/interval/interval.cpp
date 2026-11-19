// interval - the constructor of an interval, checked, as CORA's interval constructor
//
// Syntax:   Interval I(inf, sup);   Interval A = Interval::matrix(inf, sup);
// Inputs:   inf, sup - the bounds of one shape: (..., n) for a box, (..., r, c) for a matrix
// Outputs:  I - the interval {x | inf <= x <= sup}
// See also: generate_random, Zonotope::Zonotope
//
// CORA: contSet/@interval/interval.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

#include <cstdlib>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether inf <= sup everywhere (NaN bounds are left to the caller).
bool aux_ordered(const Tensor &inf, const Tensor &sup) {
    for (const double width : (sup - inf).data())
        if (width < 0) return false;
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval::Interval(Tensor inf, Tensor sup) : Interval(std::move(inf), std::move(sup), 1, {}) {}

Interval Interval::matrix(Tensor inf, Tensor sup) {
    return {std::move(inf), std::move(sup), 2, {}};
}

Interval::Interval(Tensor inf, Tensor sup, int elem_rank, EmptyMask empty)
    : ContSet(kClassId, std::move(empty)), inf_(std::move(inf)), sup_(std::move(sup)),
      elem_rank_(elem_rank) {
    detail::Entry entry;
    // tensors of one shape with the element axes
    CORA_CHECK(inf_.defined() && sup_.defined(), Error::wrong_input_in_constructor,
               [] { return std::string("Interval: the bounds must be tensors"); });
    CORA_CHECK(inf_.ndim() >= elem_rank_ && inf_.shape() == sup_.shape(),
               Error::wrong_input_in_constructor, [&] {
                   return std::string("Interval: the bounds must be ") +
                          (elem_rank_ == 1 ? "vectors (..., n)" : "matrices (..., r, c)") +
                          " of one shape, got " + shape_text(inf_.shape()) + " and " +
                          shape_text(sup_.shape());
               });
    // the bounds must be ordered
    CORA_CHECK(aux_ordered(inf_, sup_), Error::wrong_input_in_constructor, [] {
        return std::string(
            "Interval: the lower bound exceeds the upper one somewhere; give inf <= sup");
    });
}

// Private helpers: results of the shape of this interval ------------------------------------------

Interval Interval::with(Tensor inf, Tensor sup) const {
    return {std::move(inf), std::move(sup), elem_rank_, empty_};
}

Interval Interval::from_matrix(Tensor inf, Tensor sup) const {
    if (elem_rank_ == 1 && inf.shape().back() == 1)
        return {inf.squeeze(-1), sup.squeeze(-1), 1, empty_};
    return {std::move(inf), std::move(sup), 2, empty_};
}

// the element axes are the last elem_rank_, the batch axes the ones before
Shape Interval::elem_shape() const {
    Shape s;
    for (int64_t a = inf_.ndim() - elem_rank_; a < inf_.ndim(); ++a) s.push_back(inf_.shape()[a]);
    return s;
}

Shape Interval::batch_shape() const {
    Shape s;
    for (int64_t a = 0; a < inf_.ndim() - elem_rank_; ++a) s.push_back(inf_.shape()[a]);
    return s;
}

// The shift and the scaling keep the shape and the empty members.
Interval operator+(const Interval &I, const Tensor &v) { return I.with(I.inf_ + v, I.sup_ + v); }

Interval operator+(const Interval &I, double s) { return I.with(I.inf_ + s, I.sup_ + s); }

Interval operator*(double s, const Interval &I) {
    return s >= 0 ? I.with(I.inf_ * s, I.sup_ * s) : I.with(I.sup_ * s, I.inf_ * s);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
