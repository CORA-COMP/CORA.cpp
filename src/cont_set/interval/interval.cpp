// interval - the constructor of an interval, checked, as CORA's interval constructor
//
// Syntax:   Interval I(inf, sup);
// Inputs:   inf, sup - the bounds, of one shape: columns (..., n, 1), or matrices (..., n, k)
// Outputs:  I - the interval {x | inf <= x <= sup}
// See also: generate_random, Zonotope::Zonotope
//
// CORA: contSet/@interval/interval.m

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

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

Interval::Interval(Tensor inf, Tensor sup)
    : ContSet(kClassId), inf_(std::move(inf)), sup_(std::move(sup)) {
    detail::Entry entry;
    CORA_CHECK(inf_.defined() && sup_.defined(), Error::wrong_input_in_constructor,
               [] { return std::string("Interval: the bounds must be tensors"); });
    CORA_CHECK(inf_.shape().size() >= 2 && inf_.shape() == sup_.shape(),
               Error::wrong_input_in_constructor, [&] {
                   return "Interval: the bounds must be columns or matrices of one shape, got " +
                          shape_text(inf_.shape()) + " and " + shape_text(sup_.shape());
               });
    CORA_CHECK(aux_ordered(inf_, sup_), Error::wrong_input_in_constructor, [] {
        return std::string(
            "Interval: the lower bound exceeds the upper one somewhere; give inf <= sup");
    });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
