// zonotope - lean::Zonotope, a zonotope of exact values plus an error box, computed by CORALean
//
// Syntax:     lean::Zonotope Z(c, G);   Z = Z.linear_map(M).plus(Z2).reduce(1);   Z.gather();
// Parts:      nominal() the exact zonotope c + G[-1,1]^h, error() the box E that the sound float
//             operations add; together they enclose the real result (enclosure())
// Crossing:   round_to(dtype) rounds c and G and grows E; gather() returns the nominal part as
//             a cora::Zonotope and the error as a cora::Interval, exactly or throws
// See also:   lean/tensor.h, lean/oracle.h, cora/cont_set/zonotope/zonotope.h

#pragma once

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Zonotope {
  public:

    /// An exact zonotope (empty error box) in the current dtype.
    Zonotope(const cora::Tensor &c, const cora::Tensor &G);

    Zonotope(Tensor c, Tensor G, Interval E);

    /// The exact part: c and G with an empty error box.
    Zonotope nominal() const;

    /// The center (n, 1), the generators (n, h), exact, and the error box (n, 1) alone.
    const Tensor &center() const noexcept { return c_; }
    const Tensor &generators() const noexcept { return G_; }
    const Interval &error() const noexcept { return E_; }

    /// nominal() + error(): a lean zonotope with an empty error box enclosing this one.
    Zonotope enclosure() const;

    /// c and G rounded into `dtype`; the rounding errors are added to the error box.
    Zonotope round_to(const std::string &dtype, const std::string &mode = "nearest") const;

    /// M * Z for a matrix or an interval matrix.
    Zonotope linear_map(const Tensor &M) const;
    Zonotope linear_map(const Interval &M) const;

    /// The Minkowski sum with a zonotope or with a box.
    Zonotope plus(const Zonotope &Z2) const;
    Zonotope plus(const Interval &I) const;

    /// CORA's linComb with a zonotope Z2 that shares the generator factors of this one.
    Zonotope linear_combination(const Zonotope &Z2) const;

    /// Order reduction to `order` with method "girard" or "combastel".
    Zonotope reduce(int order, const std::string &method = "girard") const;

    /// The interval hull of the enclosure.
    Interval interval() const;

    /// The nominal part as a cora::Zonotope and the error as a cora::Interval (double, exact).
    std::pair<cora::Zonotope, cora::Interval> gather() const;

    Json to_json() const;
    static Zonotope from_json(const std::string &dtype, const Json &j);

  private:
    Tensor c_, G_;
    Interval E_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
