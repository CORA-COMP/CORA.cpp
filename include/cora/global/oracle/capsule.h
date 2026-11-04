// capsule - lean::Capsule, a capsule of exact values, computed by CORALean
//
// Syntax:     lean::Capsule C(cora_capsule);   lean::Capsule R = C.convex_hull(C2);   R.gather();
// Operations: one method per `capsule.*` op of the oracle, named as CORA.cpp names CORA's
//             (`mtimes` is linear_map, `convHull` convex_hull, `isIntersecting` is_intersecting);
//             k is the number of Newton steps of the square roots, which CORA has no argument for
// Parts:      the oracle's set is the capsule c, g, r plus the error box E of the sound float
//             operations; at dtype nearest the box is empty (nominal())
// Crossing:   gather() returns the nominal part as a cora::Capsule, exactly or throws
// See also:   lean/request.h, lean/interval.h, cora/cont_set/capsule/capsule.h

#pragma once

#include "cora/cont_set/capsule/capsule.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/tensor.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Capsule {
  public:
    /// An exact capsule (empty error box) in the current dtype.
    Capsule(const cora::Capsule &C);

    /// The dtype of the values.
    const std::string &dtype() const noexcept { return dtype_; }

    /// The nominal part c, g and r as the oracle returns them; the error box is not looked at.
    /// Throws if a value is not a double.
    cora::Capsule nominal() const;

    /// Whether the error box is empty (all zero).
    bool exact() const;

    /// nominal() when the error box is empty, else it throws.
    cora::Capsule gather() const;

    // Operations -------------------------------------------------------------------------------

    /// The stored center (`capsule.center`) and the box that encloses the capsule (`interval`).
    Tensor center() const;
    Interval interval() const;

    /// Whether the point p (n, 1) is in the capsule; a proof when true (`capsule.contains`).
    Certified contains(const Tensor &p) const;

    /// Whether the capsule meets C2; a proof of disjointness when false (`capsule.isIntersecting`).
    Certified is_intersecting(const Capsule &C2, int k = 100) const;

    bool is_bounded() const;
    bool is_full_dim() const;
    bool isequal(const Capsule &C2, double tol = 0) const;

    Capsule convex_hull(const Capsule &C2, int k = 100) const;
    Capsule linear_map(const Tensor &M, int k = 100) const;
    Capsule lift(int64_t N, const std::vector<int64_t> &dims) const;

    /// The hull of this capsule and M * C2 (CORA's code of enclose; it misses C2's sum).
    Capsule enclose(const Tensor &M, const Capsule &C2, int k = 100) const;

    static Capsule enclose_points(const Tensor &points, int k = 100);
    static Capsule empty(int64_t n);
    static Capsule origin(int64_t n);

    Json to_json() const;
    static Capsule from_json(const std::string &dtype, const Json &j);

  private:
    Capsule(std::string dtype, Json json) : dtype_(std::move(dtype)), json_(std::move(json)) {}

    std::string dtype_;
    Json json_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
