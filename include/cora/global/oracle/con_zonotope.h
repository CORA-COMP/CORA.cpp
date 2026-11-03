// con_zonotope - lean::ConZonotope, a constrained zonotope of exact values, computed by CORALean
//
// Syntax:     lean::ConZonotope cz(cora_cz);   lean::ConZonotope r = cz.intersect(cz2); r.gather();
// Operations: one method per `conZonotope.*` op of the oracle, named as CORA.cpp names CORA's
// (`and`
//             is intersect, `cartProd` cart_prod, `conIntersect` con_intersect, `cubMap` cub_map)
// Parts:      the oracle's set is {c + G x + E | |x + E_x|_inf <= 1, Ae x = be + be_error} with
//             the error boxes of the sound float operations; at dtype nearest they are empty
//             (nominal()), else interval() is the box the oracle proves to enclose the set
// Crossing:   gather() returns the nominal part as a cora::ConZonotope, exactly or throws
// See also:   lean/zonotope.h, lean/request.h, cora/cont_set/con_zonotope/con_zonotope.h

#pragma once

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/tensor.h"
#include "cora/global/oracle/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class ConZonotope {
  public:
    /// An exact constrained zonotope (empty error boxes) in the current dtype.
    ConZonotope(const cora::ConZonotope &cz);

    /// The dtype of the values.
    const std::string &dtype() const noexcept { return dtype_; }

    /// The nominal part c, G and the equalities (Ae, be) as the oracle returns them; throws if a
    /// value is not a double. The inequality rows and the error boxes are not looked at.
    cora::ConZonotope nominal() const;

    /// The inequality rows A x <= b of the oracle's set, as a pair (A, b): none for an input.
    std::pair<cora::Tensor, cora::Tensor> inequalities() const;

    /// nominal() when there are no inequality rows and every error box is empty, else it throws.
    cora::ConZonotope gather() const;

    /// Whether every error box is empty and there are no inequality rows.
    bool exact() const;

    /// The box the oracle proves to enclose the set (`conZonotope.interval`).
    Interval interval() const;

    // Set operations ---------------------------------------------------------------------------

    ConZonotope intersect(const ConZonotope &cz2) const;
    ConZonotope convex_hull(const ConZonotope &cz2) const;
    ConZonotope cart_prod(const ConZonotope &cz2) const;
    ConZonotope con_intersect(const ConZonotope &cz2, const Tensor &R) const;

    /// The enclosure of this set and M * this + cz2 (the oracle's `enclose`).
    ConZonotope enclose(const Tensor &M, const ConZonotope &cz2) const;

    /// The zonotope-like strips |C x - y| <= phi at the gain Lambda.
    ConZonotope intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                const Tensor &Lambda) const;

    /// The cubic map T_ij x x x; method "zeros" of compact.
    ConZonotope cub_map(const std::vector<std::vector<Tensor>> &T) const;
    ConZonotope compact(const std::string &method = "zeros", double tol = 0) const;

    static ConZonotope empty(int64_t n);

    /// A zonotope or a box as a constrained zonotope without constraints.
    static ConZonotope from_zonotope(const Zonotope &Z);
    static ConZonotope from_interval(const Interval &I);

    bool is_bounded() const;

    Json to_json() const;
    static ConZonotope from_json(const std::string &dtype, const Json &j);

  private:
    ConZonotope(std::string dtype, Json json) : dtype_(std::move(dtype)), json_(std::move(json)) {}

    std::string dtype_;
    Json json_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
