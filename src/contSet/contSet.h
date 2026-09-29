// The abstract set, as CORA's `contSet`: what every set representation can answer, so that
// code which only needs that — a specification, a plot — takes any of them.
//
// Sets are written against CoraTensor (tensor/tensor.h), so they run on every backend and
// batch wherever the backend does. A column vector is `(..., n, 1)`.

#pragma once

#include "tensor/tensor.h"

namespace cora::ct {

class Interval;

class ContSet {
  public:
    virtual ~ContSet() = default;

    /// The dimension `n`.
    virtual int64_t dim() const = 0;

    /// The centre, a column `(..., n, 1)`.
    virtual Tensor center() const = 0;

    /// `max_{x in S} dᵀx` for a column `d` `(..., n, 1)`: a `(..., 1, 1)` tensor.
    virtual Tensor support_func(const Tensor &d) const = 0;

    /// The smallest axis-aligned box containing the set.
    virtual Interval interval() const = 0;
};

} // namespace cora::ct
