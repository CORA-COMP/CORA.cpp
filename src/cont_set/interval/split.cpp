// split - a box cut in two halves at the center of one coordinate, as CORA's interval.split
//
// The center is (inf + sup) / 2, as CORALean's split up to its outward rounding of the midpoint.
//
// Syntax:   auto [I1, I2] = I.split(n);
// Inputs:   n - the coordinate to cut, 0-based; the box is a column (..., n, 1)
// Outputs:  I1, I2 - the lower half (the center as upper bound in coordinate n) and the upper half
// See also: partition, center
//
// CORA: contSet/@interval/split.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The column x with entry n replaced by that of c.
Tensor aux_replace(const Tensor &x, const Tensor &c, int64_t n) {
    const Tensor head = x.slice(-2, 0, n), tail = x.slice(-2, n + 1, x.size(-2));
    return Tensor::concat({head, c.slice(-2, n, n + 1), tail}, -2);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<Interval, Interval> Interval::split(int64_t n) const {
    detail::Entry entry;
    CORA_CHECK(inf_.size(-1) == 1, Error::wrong_value, [&] {
        return "split: the interval must not be an n-d array with n > 1, got the shape " +
               shape_text(inf_.shape());
    });
    CORA_CHECK(n >= 0 && n < dim(), Error::wrong_value, [&] {
        return "split: the coordinate must be in [0, " + std::to_string(dim()) + "), got " +
               std::to_string(n);
    });
    const Tensor m = center();
    return {Interval(inf_, aux_replace(sup_, m, n)), Interval(aux_replace(inf_, m, n), sup_)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
