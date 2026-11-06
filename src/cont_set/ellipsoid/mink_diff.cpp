// mink_diff - the Minkowski difference of an ellipsoid, as CORA's ellipsoid.minkDiff
//
// A vector shifts the center by -v. For an ellipsoid S2: two points give a point, a point minuend
// with a proper S2 the empty set, a point S2 a shift, a smaller minuend the empty set, equal shape
// matrices a point. The general case runs over directions of the equal-area partition, which
// CORA.cpp lacks: Error::not_supported. An empty S2 gives the whole space.
//
// Syntax:   Eres = E.mink_diff(v);   Eres = E.mink_diff(S2);   S = E.minkowski_difference(S2);
// Inputs:   v - column (n, 1)
//           S2 - ellipsoid
// Outputs:  Eres - {x | x + S2 in E}
// See also: plus, is_bigger
//
// CORA: contSet/@ellipsoid/minkDiff.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Ellipsoid Ellipsoid::mink_diff(const Tensor &v) const {
    detail::Entry entry;
    CORA_CHECK(is_column(v, dim()), Error::dimension_mismatch, [&] {
        return "mink_diff: the vector must be a column of the dimension " + std::to_string(dim()) +
               ", got the shape " + shape_text(v.shape());
    });
    if (priv_is_empty(*this)) return *this;
    return {Q_, q_ - v, tol_};
}

Ellipsoid Ellipsoid::mink_diff(const Ellipsoid &S2) const {
    detail::Entry entry;
    const detail::Refusal refusal = priv_pair_refusal(*this, S2, "mink_diff");
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    const double tol = std::min(tol_, S2.tol_);
    const int64_t rank1 = static_cast<int64_t>(priv_rank(Q_, tol_).item()),
                  rank2 = static_cast<int64_t>(priv_rank(S2.Q_, S2.tol_).item());
    // points: the difference of the centers, or nothing fits a proper S2 into a point
    if (rank1 == 0 && rank2 == 0) return {Q_.zeros_like(), q_ - S2.q_, tol_};
    if (rank1 == 0) return Ellipsoid::empty(dim());
    if (rank2 == 0) return {Q_, q_ - S2.q_, tol_};
    if (!is_bigger(S2)) return Ellipsoid::empty(dim());
    if (within_tol(Q_, S2.Q_, tol).min().item() != 0.0) return {Q_.zeros_like(), q_ - S2.q_, tol_};
    CORA_FAIL(Error::not_supported,
              "mink_diff: the difference of two ellipsoids needs directions of the equal-area "
              "partition of the sphere (eq_point_set)");
}

Set Ellipsoid::minkowski_difference_impl(const ContSet &S2,
                                         const std::optional<std::string> &type) const {
    const std::string mode = type.value_or("outer");
    if (mode != "outer" && mode != "inner")
        CORA_FAIL(Error::wrong_value,
                  "minkowski_difference: the type must be 'outer' or 'inner', got '" + mode + "'");
    if (S2.class_id() != ClassId::ellipsoid)
        CORA_FAIL(Error::noops,
                  "minkowski_difference is not implemented for Ellipsoid and " + S2.class_name());
    const Ellipsoid &E2 = static_cast<const Ellipsoid &>(S2);
    // subtracting the empty set leaves everything
    if (priv_is_empty(E2)) return Fullspace(dim());
    return mink_diff(E2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
