// zonotope - the constructor of a zonotope, checked, as CORA's zonotope constructor
//
// Syntax:   Zonotope Z(c, G);
// Inputs:   c - center, a column (..., n, 1);  G - generators (..., n, m), of the backend of c
// Outputs:  Z - the zonotope {c + G b | |b|_inf <= 1}
// See also: generate_random, Interval::Interval
//
// CORA: contSet/@zonotope/zonotope.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope::Zonotope(Tensor c, Tensor G) : ContSet(kClassId), c_(std::move(c)), G_(std::move(G)) {
    detail::Entry entry;
    CORA_CHECK(c_.defined() && G_.defined(), Error::wrong_input_in_constructor, [] {
        return std::string("Zonotope: the center and the generators must be tensors");
    });
    CORA_CHECK(c_.shape().size() >= 2 && c_.shape().back() == 1, Error::wrong_input_in_constructor,
               [&] {
                   return "Zonotope: the center must be a column (..., n, 1), got the shape " +
                          shape_text(c_.shape());
               });
    CORA_CHECK(G_.shape().size() >= 2 && G_.shape()[G_.shape().size() - 2] == dim(),
               Error::wrong_input_in_constructor, [&] {
                   return "Zonotope: the generators need " + std::to_string(dim()) +
                          " rows (the dimension of the center), got the shape " +
                          shape_text(G_.shape());
               });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
