// interval - the box of a polygon, as CORA's interval
//
// The hull of the pieces' boxes (CORALean's Polygon.interval): every piece's box by
// Polytope::interval, joined by CORALean's minimum and maximum; no piece gives the empty box.
//
// Syntax:   Interval I = pgon.interval();
// Inputs:   -
// Outputs:  I - the smallest box (2)
// See also: Polytope::interval, center
//
// CORA: contSet/@polygon/interval.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cont_set/polygon/private/priv.h"
#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Polygon::interval_impl() const {
    if (pieces_.empty()) return Interval::empty(2);
    const Interval last = pieces_.back().interval();
    std::vector<double> lo = last.inf().data(), hi = last.sup().data();
    // CORALean's intervalOfList: box_0 or (box_1 or (... or box_last)), from the last piece
    for (std::size_t i = pieces_.size() - 1; i-- > 0;) {
        const Interval box = pieces_[i].interval();
        const std::vector<double> l = box.inf().data(), h = box.sup().data();
        for (int j = 0; j < 2; ++j) {
            lo[j] = priv_minimum(l[j], lo[j]);
            hi[j] = priv_maximum(h[j], hi[j]);
        }
    }
    return Interval(Tensor::like(like_, lo, {2}), Tensor::like(like_, hi, {2}));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
