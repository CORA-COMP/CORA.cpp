// zonotope - the constructors of a zonotope, checked, as CORA's zonotope constructor
//
// Syntax:   Zonotope Z(c, G);   Zonotope A = Zonotope::matrix(C, G);
// Inputs:   c - center (batch..., n) or a column (batch..., n, 1); C (batch..., r, c) of matrices
//           G - generators (batch..., h, n), one per row; (batch..., h, r, c), of the backend of c
// Outputs:  Z - the zonotope {c + sum_i b_i G_i | |b|_inf <= 1}
// See also: generate_random, Interval::Interval
//
// CORA: contSet/@zonotope/zonotope.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Tobias Ladner
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether G (batch..., h, elem...) fits the center c (batch..., elem...).
bool aux_fits(const Shape &c, const Shape &G, int64_t elem_rank) {
    if (G.size() != c.size() + 1 || c.size() < elem_rank) return false;
    const int64_t batch = c.size() - elem_rank;
    for (int64_t i = 0; i < batch; ++i)
        if (c[i] != G[i]) return false;
    for (int64_t i = batch; i < c.size(); ++i)
        if (c[i] != G[i + 1]) return false;
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope::Zonotope(Tensor c, Tensor G) : Zonotope(std::move(c), std::move(G), 1) {}

Zonotope Zonotope::matrix(Tensor C, Tensor G) { return {std::move(C), std::move(G), 2}; }

Zonotope::Zonotope(Tensor c, Tensor G, int64_t elem_rank)
    : ContSet(kClassId), c_(std::move(c)), G_(std::move(G)), elem_rank_(elem_rank) {
    detail::Entry entry;
    CORA_CHECK(c_.defined() && G_.defined(), Error::wrong_input_in_constructor, [] {
        return std::string("Zonotope: the center and the generators must be tensors");
    });
    // a center given as a column (batch..., n, 1), as Tensor({...}) makes one, is the vector (n)
    if (elem_rank_ == 1 && c_.ndim() >= 2 && c_.ndim() == G_.ndim() && c_.size(-1) == 1 &&
        c_.size(-2) == G_.size(-1))
        c_ = c_.squeeze(-1);

    CORA_CHECK(c_.ndim() >= elem_rank_, Error::wrong_input_in_constructor, [&] {
        return "Zonotope: the center must be (batch..., elem...) with " +
               std::to_string(elem_rank_) + " element dims, got the shape " +
               shape_text(c_.shape());
    });
    CORA_CHECK(
        aux_fits(c_.shape(), G_.shape(), elem_rank_), Error::wrong_input_in_constructor, [&] {
            return "Zonotope: the generators must be (batch..., h, elem...) for the center " +
                   shape_text(c_.shape()) + ", one generator per row, got the shape " +
                   shape_text(G_.shape());
        });
}

// the element and the batch dims of the center
Shape Zonotope::elem_shape() const {
    const Shape &s = c_.shape();
    return Shape(std::vector<int64_t>(s.begin() + (s.size() - elem_rank_), s.end()));
}

Shape Zonotope::batch_shape() const {
    const Shape &s = c_.shape();
    return Shape(std::vector<int64_t>(s.begin(), s.begin() + (s.size() - elem_rank_)));
}

// a matrix zonotope has as many dimensions as entries
int64_t Zonotope::dim_impl() const { return elem_shape().numel(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
