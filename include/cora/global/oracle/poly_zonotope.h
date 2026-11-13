// poly_zonotope - lean::PolyZonotope, a polynomial zonotope computed by CORALean
//
// Syntax:     lean::PolyZonotope P(pZ);   P = P.plus(Z).linear_map(M).reduce(2);   P.gather();
// Operations: one method per `polyZonotope.*` op of the oracle, named as CORA.cpp names CORA's;
//             the constructor of CORA is `construct`, which takes the columns as given
// Parts:      the center c, the dependent generators G with the exponents (one column per
//             generator, as cora::Exponents), the independent generators GI, the identifiers and
//             the error box E that the sound float operations add; gather() returns the exact
//             part as a cora::PolyZonotope, with every column as the oracle returns it
// See also:   lean/request.h, cora/cont_set/poly_zonotope/poly_zonotope.h
//
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/tensor.h"
#include "cora/global/oracle/zonotope.h"

#include <cstdint>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class PolyZonotope {
  public:
    /// The exact set (empty error box) of a cora::PolyZonotope, in the current dtype.
    PolyZonotope(const cora::PolyZonotope &P);

    PolyZonotope(Tensor c, Tensor G, Tensor GI, Interval E, cora::Exponents Exp,
                 std::vector<int64_t> id);

    /// CORA's constructor on the columns as given: equal exponent columns added, zero generators
    /// dropped, in the order of CORA's hash.
    static PolyZonotope construct(const cora::Tensor &c, const cora::Tensor &G,
                                  const cora::Tensor &GI, const cora::Exponents &Exp,
                                  const std::vector<int64_t> &id);

    /// The polynomial zonotope of a zonotope or a box (identity exponents, then the constructor).
    static PolyZonotope from_zonotope(const cora::Zonotope &Z);
    static PolyZonotope from_interval(const cora::Interval &I);

    const Tensor &center() const noexcept { return c_; }
    const Tensor &generators() const noexcept { return G_; }
    const Tensor &independent_generators() const noexcept { return GI_; }
    const Interval &error() const noexcept { return E_; }
    const cora::Exponents &exponents() const noexcept { return Exp_; }
    const std::vector<int64_t> &ids() const noexcept { return id_; }

    // Operations -------------------------------------------------------------------------------

    /// The sum with a polynomial zonotope (its factors independent), a zonotope, a box or a point.
    PolyZonotope plus(const PolyZonotope &Q) const;
    PolyZonotope plus(const cora::Zonotope &Z) const;
    PolyZonotope plus(const cora::Interval &I) const;
    PolyZonotope plus(const cora::Tensor &v) const;

    /// M * P for a matrix M.
    PolyZonotope linear_map(const cora::Tensor &M) const;

    /// CORA's reduce(P, 'girard', order) and enclose(P, Q).
    PolyZonotope reduce(int order) const;
    PolyZonotope enclose(const PolyZonotope &Q) const;

    /// CORA's zonotope(P) and interval(P), and the dimension.
    Zonotope zonotope() const;
    Interval interval() const;
    int64_t dim() const;

    /// The exact part as a cora::PolyZonotope, every column kept as it is; throws if the error
    /// box is not empty.
    cora::PolyZonotope gather() const;

    Json to_json() const;
    static PolyZonotope from_json(const std::string &dtype, const Json &j);

  private:
    Tensor c_, G_, GI_;
    Interval E_;
    cora::Exponents Exp_;
    std::vector<int64_t> id_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
