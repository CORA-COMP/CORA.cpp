// interval - the class of intervals I = {x | inf <= x <= sup}, as CORA's interval
//
// A box with column bounds (..., n, 1). As in CORA the bounds may be matrices (..., n, n): an
// interval matrix is an Interval too, and Zonotope::mtimes takes it.
//
// Syntax:     Interval I(inf, sup);   Interval I = Interval::generateRandom(n, rng);
// Operations: supportFunc, mtimes, randPoint, contains, generateRandom (one file each);
//             center, rad, plus (below)
// See also:   contSet/contSet.h, contSet/zonotope/zonotope.h

#pragma once

#include "contSet/contSet.h"

namespace cora::ct {

class Interval : public ContSet {
  public:
    /// The bounds: columns (..., n, 1) for a box, matrices (..., n, n) for an interval matrix.
    Tensor inf, sup;

    Interval(Tensor inf, Tensor sup) : inf(std::move(inf)), sup(std::move(sup)) {}

    /// A random box in n dimensions (CORA: center U[-2, 2], radius R/2*u, R ~ U[0, 10], u ~ U[0, 1]).
    static Interval generateRandom(int64_t n, Rng &rng);

    /// The dimension: the rows of the bounds.
    int64_t dim() const override { return inf.shape()[inf.shape().size() - 2]; }

    /// (inf + sup) / 2, elementwise, of the shape of the bounds.
    Tensor center() const override { return (inf + sup) * 0.5; }

    /// (sup - inf) / 2, elementwise, of the shape of the bounds.
    Tensor rad() const { return (sup - inf) * 0.5; }

    /// max_{x in I} d'x = d'c + |d|'r; d is a column (..., n, 1).
    Tensor supportFunc(const Tensor &d) const override;

    /// An interval is its own interval hull.
    Interval interval() const override { return *this; }

    /// N points uniform in the box, columns (..., n, N).
    Tensor randPoint(int64_t N, Rng &rng) const override;

    /// M * I enclosed by its interval hull: the center maps by M, the radius by |M|.
    Interval mtimes(const Tensor &M) const;

    /// The Minkowski sum: the bounds add.
    Interval plus(const Interval &I2) const { return {inf + I2.inf, sup + I2.sup}; }

    /// Whether the point p (..., n, 1) is in the box, boundary included; for a batch, all points.
    bool contains(const Tensor &p) const;
};

} // namespace cora::ct
