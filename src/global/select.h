// select - the matrix that picks dimensions, shared by the projections of the sets
//
// project(dims) of every set is a product with this matrix, so it runs on every backend.

#pragma once

#include "tensor/tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The (k, n) matrix P with P x = (x[dims[0]], ..., x[dims[k-1]]), on the backend and device of
/// `like`; throws if a dimension is out of 0..n-1.
inline Tensor selectDims(const Tensor &like, const std::vector<int64_t> &dims, int64_t n) {
    const int64_t k = static_cast<int64_t>(dims.size());
    std::vector<double> data(static_cast<std::size_t>(k * n), 0.0);
    for (int64_t i = 0; i < k; ++i) {
        if (dims[i] < 0 || dims[i] >= n)
            throw std::invalid_argument("project: the dimension " + std::to_string(dims[i]) +
                                        " is outside 0.." + std::to_string(n - 1) +
                                        " (dimensions are 0-based)");
        data[static_cast<std::size_t>(i * n + dims[i])] = 1.0;
    }
    return Tensor::like(like, data, {k, n});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
