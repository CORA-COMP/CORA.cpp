// zonotope - lean::Zonotope, a zonotope of exact values plus an error box, computed by CORALean
//
// Syntax:     lean::Zonotope Z(c, G);   Z = Z.mtimes(M).plus(Z2).reduce(1);   Z.gather();
// Parts:      nominal() the exact zonotope c + G[-1,1]^h, error() the box E that the sound float
//             operations add; together they enclose the real result (enclosure())
// Crossing:   roundTo(dtype) rounds c and G and grows E; gather() returns the nominal part as
//             a cora::Zonotope and the error as a cora::Interval, exactly or throws
// See also:   lean/tensor.h, lean/oracle.h, contSet/zonotope/zonotope.h

#pragma once

#include "contSet/zonotope/zonotope.h"
#include "lean/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Zonotope {
  public:
    Tensor c, G;  // center (n, 1) and generators (n, h), exact
    Interval E;   // the error box (n, 1)

    /// An exact zonotope (empty error box) in the current dtype.
    Zonotope(const cora::Tensor &c, const cora::Tensor &G);

    Zonotope(Tensor c, Tensor G, Interval E);

    /// The exact part: c and G with an empty error box.
    Zonotope nominal() const;

    /// The error box alone.
    const Interval &error() const { return E; }

    /// nominal() + error(): a lean zonotope with an empty error box enclosing this one.
    Zonotope enclosure() const;

    /// c and G rounded into `dtype`; the rounding errors are added to the error box.
    Zonotope roundTo(const std::string &dtype, const std::string &mode = "nearest") const;

    /// M * Z for a matrix or an interval matrix.
    Zonotope mtimes(const Tensor &M) const;
    Zonotope mtimes(const Interval &M) const;

    /// The Minkowski sum with a zonotope or with a box.
    Zonotope plus(const Zonotope &Z2) const;
    Zonotope plus(const Interval &I) const;

    /// CORA's linComb with a zonotope Z2 that shares the generator factors of this one.
    Zonotope linComb(const Zonotope &Z2) const;

    /// Order reduction to `order` with method "girard" or "combastel".
    Zonotope reduce(int order, const std::string &method = "girard") const;

    /// The interval hull of the enclosure.
    Interval interval() const;

    /// The nominal part as a cora::Zonotope and the error as a cora::Interval (double, exact).
    std::pair<cora::Zonotope, cora::Interval> gather() const;

    Json toJson() const;
    static Zonotope fromJson(const std::string &dtype, const Json &j);
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
