// contSet - the abstract set, as CORA's contSet
//
// Interval and Zonotope derive from it; code that needs only what it offers (a specification,
// a plot, a simulation) takes any set. Sets run on every CoraTensor backend and batch wherever
// the backend does; a vector is a column (..., n, 1). Each operation has its own file in the
// set's folder, named as in MATLAB CORA, with its auxiliary functions at the bottom.

#pragma once

#include "tensor/tensor.h"

#include <array>
#include <memory>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {
class Rng;
}

namespace cora::ct {

class Interval;

/// A point of the plane, and a polygon as its vertices in counter-clockwise order.
using Point = std::array<double, 2>;
using Polygon = std::vector<Point>;

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

    /// The projection onto the dimensions `dims` (0-based), in that order.
    virtual std::unique_ptr<ContSet> project(const std::vector<int64_t> &dims) const = 0;

    /// The vertices of a two-dimensional set, one polygon per batch member.
    virtual std::vector<Polygon> vertices() const = 0;
};

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
