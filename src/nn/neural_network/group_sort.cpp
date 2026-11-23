// group_sort - CORA's group sort layer on points: groups of neurons sorted ascending
//
// A group of g consecutive neurons is sorted ascending; a layer of n neurons has ceil(n / g)
// groups, the last one shorter if g does not divide n (CORA pads the layer with inf, sorts, and
// cuts the padding, so the padding stays last). The gradient of a sort is a permutation: the
// entry of each sorted position returns to where it came from (a stable sort, as MATLAB's: equal
// values keep their order). CORA's backpropNumeric reads grad_out(perm) instead, which is the
// inverse permutation for groups of two only; this is the permutation's transpose.
//
// Syntax:   group_sort() - the functions, e.g.
//           Tensor y = priv_group_sort(x, g);   Tensor gx = priv_group_sort_t(x, gy, g);
// Inputs:   x - rows (..., n)
//           gy - the gradient at the output (..., n)
//           g - the group size
// Outputs:  y - the sorted rows
//           gx - the gradient at the input
// See also: cora/nn/neural_network/neural_network.h, evaluate.cpp, backprop.cpp
// References: none (CORA's layer, no paper's algorithm)
//
// CORA: nn/layers/nonlinear/nnGroupSortLayer.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "nn/neural_network/private/priv_layers.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows padded to a multiple of g with `fill`, as (..., groups, g).
Tensor aux_groups(const Tensor &x, int64_t g, double fill) {
    const int64_t n = x.shape().back();
    const int64_t pad = (g - n % g) % g;
    Tensor padded = x;
    if (pad > 0) {
        std::vector<int64_t> shape = x.shape();
        shape.back() = pad;
        padded = Tensor::concat({x, x.full_like(shape, fill)}, -1);
    }
    std::vector<int64_t> shape = x.shape();
    shape.back() = (n + pad) / g;
    shape.push_back(g);
    return padded.reshape(shape);
}

/// The groups (..., groups, g) as rows of n neurons again.
Tensor aux_rows(const Tensor &groups, int64_t n) {
    std::vector<int64_t> shape = groups.shape();
    shape.pop_back();
    shape.back() = groups.shape()[groups.ndim() - 2] * groups.shape().back();
    return groups.reshape(shape).slice(-1, 0, n);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor priv_group_sort(const Tensor &x, int64_t group) {
    detail::Entry entry;
    CORA_CHECK(group >= 1, Error::wrong_value, [&] {
        return "group_sort: the group size must be at least 1, got " + std::to_string(group);
    });
    return aux_rows(aux_groups(x, group, std::numeric_limits<double>::infinity()).sort(-1),
                    x.shape().back());
}

Tensor priv_group_sort_t(const Tensor &x, const Tensor &grad, int64_t group) {
    detail::Entry entry;
    CORA_CHECK(group >= 1, Error::wrong_value, [&] {
        return "group_sort: the group size must be at least 1, got " + std::to_string(group);
    });
    // the order of each group; the gradient of position j returns to the input it came from
    const Tensor order = aux_groups(x, group, std::numeric_limits<double>::infinity()).argsort(-1);
    const Tensor back = aux_groups(grad, group, 0.0).take_along_axis(order.argsort(-1), -1);
    return aux_rows(back, x.shape().back());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
