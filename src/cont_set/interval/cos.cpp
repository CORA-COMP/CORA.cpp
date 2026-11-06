// cos - the cosine of a box, as CORA's interval.cos
//
// maxcos shifts the box by the multiple of 2 pi that puts its lower bound in (-2 pi, 0], gives 1
// if the shifted upper bound is positive and else the larger cosine of the two bounds. The
// multiples of 2 pi are computed in floating point and nothing rounds outward, as in CORA.
//
// Syntax:   Interval R = I.cos();
// Inputs:   -
// Outputs:  R - [-maxcos(I - pi), maxcos(I)]
// See also: cosh, exp
//
// CORA: contSet/@interval/cos.m
// CORA authors: Matthias Althoff, Dmitry Grebenyuk, Adrian Kulmburg
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kPi = 3.14159265358979323846;

/// The maximum of the cosine on [lb, ub]: CORA's aux_maxcos.
Tensor aux_max_cos(const Tensor &lb, const Tensor &ub) {
    const Tensor k = (lb / (2 * kPi)).ceil();
    const Tensor a = lb - k * (2 * kPi), b = ub - k * (2 * kPi);
    return b.sign().maximum(a.cos().maximum(b.cos()));
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval Interval::cos() const {
    detail::Entry entry;
    return {aux_max_cos(inf_ - kPi, sup_ - kPi).negative(), aux_max_cos(inf_, sup_)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
