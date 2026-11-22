// plus - the Minkowski sum, as CORA's zonotope.plus
//
// Syntax:   Zres = Z.plus(Z2);
// Inputs:   Z2 - zonotope of the same dimension
// Outputs:  Zres - Z + Z2: center c + c2, the generators of both
// See also: linear_map, linear_combination
// References:
//    [1] A. Girard. "Reachability of uncertain linear systems using zonotopes", HSCC 2005, LNCS
//        3414, pp. 291-305.
//
// CORA: contSet/@zonotope/plus.m
// CORA: matrixSet/@matZonotope/plus.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether v is a point (..., elem...) of Z: its last axes are the shape of Z's elements.
bool aux_is_point(const Tensor &v, const Zonotope &Z) {
    if (v.numel() == 1) return true; // one number for every entry
    const Shape elem = Z.elem_shape();
    if (v.ndim() < elem.size()) return false;
    for (int64_t i = 0; i < elem.size(); ++i)
        if (v.size(int(v.ndim() - elem.size() + i)) != elem[i]) return false;
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::plus(const Zonotope &Z2) const {
    detail::Entry entry;
    CORA_CHECK(Z2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "plus: the zonotopes differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(Z2.dim());
    });
    CORA_CHECK(Z2.elem_rank() == elem_rank_ && Z2.elem_shape() == elem_shape(),
               Error::dimension_mismatch, [&] {
                   return "plus: the elements differ in shape, " + shape_text(elem_shape()) +
                          " and " + shape_text(Z2.elem_shape());
               });
    // the batch dims broadcast; the generators join along their own axis
    const Tensor c = c_ + Z2.center();
    const int64_t axis = c.ndim() - elem_rank_;
    const auto widen = [&](const Tensor &G) {
        Shape s;
        for (int64_t i = 0; i < axis; ++i) s.push_back(c.shape()[i]);
        for (int64_t i = G.ndim() - elem_rank_ - 1; i < G.ndim(); ++i) s.push_back(G.shape()[i]);
        return G.shape() == s ? G : G.broadcast_to(s);
    };
    return {c, Tensor::concat({widen(G_), widen(Z2.generators())}, int(axis)), elem_rank_};
}


Zonotope Zonotope::plus(const Interval &I) const {
    detail::Entry entry;
    CORA_CHECK(
        elem_rank_ == 1 && I.dim() == dim() && I.elem_rank() == 1, Error::dimension_mismatch, [&] {
            return "plus: the interval must be a box of the dimension " + std::to_string(dim()) +
                   " of the zonotope, got bounds of the shape " + shape_text(I.inf().shape());
        });
    // the box as a zonotope: its center, and its radius as one generator per axis
    return {c_ + I.center(), Tensor::concat({G_, I.half_width().expand_dims(-1).diag()}, -2)};
}

Set Zonotope::translate_impl(const Tensor &v) const { return *this + v; }

Zonotope operator+(const Zonotope &Z, const Tensor &v) {
    detail::Entry entry;
    CORA_CHECK(aux_is_point(v, Z), Error::dimension_mismatch, [&] {
        return std::string("plus: the point must be (..., ") +
               shape_text(Z.elem_shape()).substr(1) + ", got the shape " + shape_text(v.shape());
    });
    return {Z.c_ + v, Z.G_, Z.elem_rank_};
}

Zonotope operator-(const Zonotope &Z, const Tensor &v) {
    detail::Entry entry;
    CORA_CHECK(aux_is_point(v, Z), Error::dimension_mismatch, [&] {
        return std::string("minus: the point must be (..., ") +
               shape_text(Z.elem_shape()).substr(1) + ", got the shape " + shape_text(v.shape());
    });
    return {Z.c_ - v, Z.G_, Z.elem_rank_};
}

Zonotope operator-(const Zonotope &Z) { return {Z.c_ * -1.0, Z.G_ * -1.0, Z.elem_rank_}; }


} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
