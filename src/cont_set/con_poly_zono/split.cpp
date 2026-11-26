// split - the halves of a constrained polynomial zonotope along a coordinate, as CORA's split
//
// The box I (default: interval()) is cut at the middle of side dim, (inf + sup) / 2 as CORALean's
// splitAt; each half is a zonotope of one generator per side (CORALean's Interval.zonotope),
// made a constrained polynomial zonotope and intersected with the set (CORALean's splitWith).
//
// Syntax:   [S1, S2] = cPZ.split(dim);   [S1, S2] = cPZ.split(dim, I);
// Inputs:   dim - the coordinate (0-based)
//           I - a box that holds the set
// Outputs:  S1, S2 - the lower and the upper half
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/split.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"
#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<ConPolyZono, ConPolyZono> ConPolyZono::split(int64_t d,
                                                       const std::optional<Interval> &I) const {
    detail::Entry entry;
    CORA_CHECK(d >= 0 && d < dim(), Error::wrong_value, [&] {
        return "split: the coordinate " + std::to_string(d) + " is not one of the " +
               std::to_string(dim()) + " dimensions";
    });
    const Interval box = I ? *I : interval();
    // the middle of side d, as the new upper and lower end
    const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
    const double mid = (lo[std::size_t(d)] + hi[std::size_t(d)]) * 0.5;
    std::vector<double> lo2 = lo, hi1 = hi;
    hi1[std::size_t(d)] = mid;
    lo2[std::size_t(d)] = mid;
    const Interval I1(box.inf(), Tensor::like(box.sup(), hi1, {dim()}));
    const Interval I2(Tensor::like(box.inf(), lo2, {dim()}), box.sup());
    return {intersect(ConPolyZono(I1)), intersect(ConPolyZono(I2))};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
