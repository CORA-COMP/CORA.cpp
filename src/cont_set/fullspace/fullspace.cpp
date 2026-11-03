// fullspace - the constructor of the whole space, checked, and what ContSet needs of every class
//
// Syntax:   Fullspace F(n);   F.support_function(d);   F.rand_point(N, rng);   F.project(dims);
// Inputs:   n - the dimension, an integer >= 0
// Outputs:  F - the set R^n. support_function is Inf along a nonzero direction and 0 along zero,
//           rand_point draws standard normal points (Error::not_supported in R^0), project is the
//           whole space of the kept dimensions, the vertices are the corners of the unbounded box
//           and display names the class and the dimension
// See also: inf, interval
//
// CORA: contSet/@fullspace/fullspace.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/rng.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace::Fullspace(int64_t n) : DimensionSet(kClassId, n) {}

Tensor Fullspace::support_function_impl(const Tensor &d) const {
    // max d'x over R^n is Inf unless d = 0; the largest |d_i| tells which
    const Tensor largest = d.abs().max(-2, true);
    const double kInf = std::numeric_limits<double>::infinity();
    return Tensor::where(largest.equal(largest.zeros_like()), largest,
                         largest.full_like(largest.shape(), kInf));
}

CORA_FROM_NUMBERS Tensor Fullspace::rand_point_impl(int64_t N, Rng &rng) const {
    if (n_ == 0) CORA_FAIL(Error::not_supported, "rand_point: sampling of R^0 is not supported");
    std::vector<double> points(n_ * N);
    rng.normal(points.data(), points.size(), 1.0);
    return Tensor::from_data(points, {n_, N});
}

// Projecting the whole space onto some dimensions gives the whole space of those.
Set Fullspace::project_impl(const std::vector<int64_t> &dims) const {
    return Fullspace(static_cast<int64_t>(dims.size()));
}

// The corners are those of the unbounded box, as CORA takes the vertices of interval(fs).
std::vector<Polygon> Fullspace::vertices_impl() const { return interval_impl().vertices(); }

std::vector<std::vector<Point3>> Fullspace::vertices_3d_impl() const {
    return interval_impl().vertices_3d();
}

std::string Fullspace::display_impl() const {
    return "fullspace\n- dimension: " + std::to_string(n_) + "\n";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
