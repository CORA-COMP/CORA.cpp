// zonotope - a zonotope that holds or lies in an ellipsoid, as CORA's ellipsoid.zonotope
//
// "outer:box": the parallelotope q + sqrt(Q) [-1, 1]^n that holds the ellipsoid, on the range of Q
// for a degenerate one. "inner:box": the one scaled by 1 / sqrt(rank) that lies in it. The norm
// modes need the equal-area partition of the sphere and the zonotope norm: Error::not_supported.
//
// Syntax:   Z = E.zonotope();   Z = E.zonotope(mode, number_of_generators);
// Inputs:   mode - "outer:box" (default), "inner:box", or a norm mode (not supported)
//           number_of_generators - positive integer, for the norm modes
// Outputs:  Z - the zonotope
// See also: interval, Capsule::zonotope
//
// CORA: contSet/@ellipsoid/zonotope.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Ellipsoid::zonotope(const std::string &mode,
                             std::optional<int64_t> number_of_generators) const {
    detail::Entry entry;
    CORA_CHECK(mode == "outer:box" || mode == "outer:norm" || mode == "outer:norm_bnd" ||
                   mode == "inner:box" || mode == "inner:norm" || mode == "inner:norm_bnd",
               Error::wrong_value, [&] { return "zonotope: unknown mode '" + mode + "'"; });
    CORA_CHECK(!number_of_generators || *number_of_generators >= 1, Error::wrong_value,
               [] { return std::string("zonotope: the number of generators must be positive"); });
    CORA_CHECK(!priv_is_empty(*this), Error::empty_set, [] {
        return std::string("zonotope: the ellipsoid is empty and CORA.cpp has no empty zonotope");
    });
    CORA_CHECK(mode == "outer:box" || mode == "inner:box", Error::not_supported, [&] {
        return "zonotope: the mode '" + mode + "' needs the equal-area partition of the sphere";
    });
    CORA_CHECK(Q_.ndim() == 2, Error::not_supported,
               [] { return std::string("zonotope: it supports a single ellipsoid, not a batch"); });
    const Tensor G = priv_range_factor(*this);
    const int64_t rank = G.shape().back(), n = dim();
    if (rank == 0) return {q_, G};
    // a full-rank ellipsoid uses the symmetric root U sqrt(D) U'; a degenerate one the range factor
    Tensor generators = G;
    if (rank == n) generators = G.matmul(Q_.svd()[0].matrix_transpose());
    if (mode == "inner:box") generators = generators / std::sqrt(static_cast<double>(rank));
    return {q_, generators};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
