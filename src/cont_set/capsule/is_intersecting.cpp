// is_intersecting - whether a capsule meets points or a capsule
//
// Syntax:   res = C.is_intersecting(S);   res = C.is_intersecting(S, tol);   C.is_intersecting(p)
// Inputs:   S - capsule of the dimension of C;  p - points (..., n, N);  tol - the distance of the
//           axes may exceed the sum of the radii by tol (as within_tol); default: the repository
//           tolerance (docs/architecture.md)
// Outputs:  res - true if the least distance of the two axes is at most r + S.r; for points, if
//           all are in C. The distance is exact (a closed form for two segments); CORA's
//           quadratic program has a Hessian a quarter of the right one, which pushes the minimizer
//           to the box and misses intersections (docs/cora-differences.md). An empty capsule
//           meets nothing. Of a batch: every member
// See also: contains
//
// CORA: contSet/@capsule/isIntersecting_.m

#include "cont_set/capsule/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Capsule::is_intersecting(const Tensor &p, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "is_intersecting");
    const detail::Refusal refusal = priv_points_refusal(*this, p);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    // the points of the numeric case are CORA's call of contains_
    return contains(p, tol);
}

bool Capsule::is_intersecting(const Capsule &S, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "is_intersecting");
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "is_intersecting: the capsules differ in dimension, " + std::to_string(dim()) +
               " and " + std::to_string(S.dim());
    });
    const Tensor distance = priv_segment_distance(c_, g_, S.c_, S.g_);
    const Tensor meet = priv_at_most(distance, r_.add(S.r_), tol);
    const Tensor present = (1.0 - priv_empty_mask(r_)).multiply(1.0 - priv_empty_mask(S.r_));
    return priv_all(present.multiply(meet));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
