// polytope - lean::Polytope, a polytope of exact values in either form, computed by CORALean
//
// Syntax:     lean::Polytope P(cora_polytope);   lean::Polytope R = P.intersect(P2); R.gather();
// Operations: one method per `polytope.*` op of the oracle, named as CORA.cpp names CORA's (`and`
//             is intersect, `cartProd` cart_prod, `convHull` convex_hull, `conZonotope`
//             con_zonotope);
//             halfspace form: Inf, empty, and, cartProd, contains; vertex form: and, interval,
//             center, convHull, enclosePoints, conZonotope
// Parts:      the halfspace form is {A x <= b, Ae x = be} widened by error boxes of the sound float
//             operations, the vertex form {V}; at dtype nearest the errors are empty (exact())
// Crossing:   gather() returns the nominal part as a cora::Polytope, exactly or throws
// See also:   lean/con_zonotope.h, lean/request.h, cora/cont_set/polytope/polytope.h

#pragma once

#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/oracle/con_zonotope.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/tensor.h"
#include "cora/global/oracle/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Polytope {
  public:
    /// An exact polytope (empty error boxes) in the current dtype: its halfspaces if it has them,
    /// else its vertices.
    Polytope(const cora::Polytope &P);

    const std::string &dtype() const noexcept { return dtype_; }

    /// Whether the set is given by vertices.
    bool vertices() const { return json_.find("V") != nullptr; }

    /// Whether every error box is empty.
    bool exact() const;

    /// The nominal part as the oracle returns it; throws if a value is not a double.
    cora::Polytope nominal() const;

    /// nominal() when every error box is empty, else it throws.
    cora::Polytope gather() const;

    // Halfspace form ---------------------------------------------------------------------------

    static Polytope inf(int64_t n);
    static Polytope empty(int64_t n);
    Polytope cart_prod(const Polytope &P2) const;

    /// The box I or the zonotope Z inside the halfspaces (the oracle's `contains`, a proof).
    bool contains(const Interval &I, double tol = 0) const;
    bool contains(const Zonotope &Z, double tol = 0) const;

    // Both forms -------------------------------------------------------------------------------

    Polytope intersect(const Polytope &P2) const;

    // Vertex form ------------------------------------------------------------------------------

    /// The box of the vertices (`interval`; `box` is the same).
    Interval interval() const;

    /// The box that holds the mean of the vertices (`center`).
    Interval center() const;

    Polytope convex_hull(const Polytope &P2) const;
    ConZonotope con_zonotope() const;
    static Polytope enclose_points(const Tensor &points);

    /// A box as a polytope (`interval.polytope`).
    static Polytope from_interval(const Interval &I);

    Json to_json() const { return json_; }
    static Polytope from_json(const std::string &dtype, const Json &j) { return {dtype, j}; }

  private:
    Polytope(std::string dtype, Json json) : dtype_(std::move(dtype)), json_(std::move(json)) {}

    std::string dtype_;
    Json json_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
