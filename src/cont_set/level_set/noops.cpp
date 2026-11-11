// noops - the operations that a level set does not have, as CORA's contSet raises noops
//
// A level set has no center, support function, interval, random points, projection or vertices:
// the constraints are arbitrary expressions. Each raises Error::noops, as CORA does.
//
// Syntax:   ls.center();   ls.interval();   // each ends in noops("center", ls), noops(...)
// Inputs:   -
// Outputs:  -
// See also: contains, is_intersecting
//
// CORA.cpp only: CORA's contSet raises noops for each of them; the class implements nothing
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/level_set/level_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor LevelSet::center_impl() const { priv_noops("center", *this); }

Tensor LevelSet::support_function_impl(const Tensor &) const {
    priv_noops("support_function", *this);
}

Interval LevelSet::interval_impl() const { priv_noops("interval", *this); }

Tensor LevelSet::rand_point_impl(int64_t, Rng &) const { priv_noops("rand_point", *this); }

Set LevelSet::project_impl(const std::vector<int64_t> &) const { priv_noops("project", *this); }

std::vector<Polygon> LevelSet::vertices_impl() const { priv_noops("vertices", *this); }

std::vector<std::vector<Point3>> LevelSet::vertices_3d_impl() const {
    priv_noops("vertices_3d", *this);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
