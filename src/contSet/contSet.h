// contSet - the abstract set, as CORA's contSet
//
// Interval and Zonotope derive from it; code that needs only what it offers (a specification,
// a plot, a simulation) takes any set. Sets run on every CoraTensor backend and batch wherever
// the backend does; a vector is a column (..., n, 1). Each operation has its own file in the
// set's folder, named as in MATLAB CORA, with its auxiliary functions at the bottom.

#pragma once

#include "tensor/tensor.h"

namespace cora {
class Rng;
}

namespace cora::ct {

class Interval;

class ContSet {
  public:
    virtual ~ContSet() = default;

    /// The dimension n.
    virtual int64_t dim() const = 0;

    /// The center, a column (..., n, 1).
    virtual Tensor center() const = 0;

    /// max_{x in S} d'x for a direction d (..., n, 1); returns (..., 1, 1).
    virtual Tensor supportFunc(const Tensor &d) const = 0;

    /// The smallest axis-aligned box containing the set.
    virtual Interval interval() const = 0;

    /// N random points of the set as columns (..., n, N), drawn on the host with rng.
    virtual Tensor randPoint(int64_t N, Rng &rng) const = 0;
};

} // namespace cora::ct
