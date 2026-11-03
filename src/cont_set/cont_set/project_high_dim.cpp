// project_high_dim - a set placed into a higher-dimensional space, as CORA's contSet.projectHighDim
//
// Syntax:   R = S.project_high_dim(N, proj);
// Inputs:   S - set of dimension n;  N - dimension of the space, N >= n;  proj - the N-dimensional
//           coordinate (0-based) of each of the n coordinates of S, n entries in [0, N)
// Outputs:  R - the set with coordinate proj[j] equal to the j-th coordinate of S and 0 elsewhere;
//           the generic form is the matrix I(:, proj) applied to S
// See also: project, linear_map, Interval::project_high_dim
//
// CORA: contSet/@contSet/projectHighDim.m
// CORA: contSet/@contSet/projectHighDim_.m

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


namespace detail {

Refusal project_high_dim_refusal(int64_t n, int64_t N, const std::vector<int64_t> &proj,
                                 const char *operation) {
    const std::string name = std::string(operation) + ": ";
    // CORA's checks, in its order: the size of the space, the number and the range of coordinates
    if (N < n)
        return std::pair{Error::wrong_value,
                         name + "the space of " + std::to_string(N) +
                             " dimensions must not be smaller than the set of " +
                             std::to_string(n)};
    if (static_cast<int64_t>(proj.size()) != n)
        return std::pair{Error::wrong_value,
                         name + "one coordinate per dimension of the set is needed, "
                         "got " + std::to_string(proj.size()) + " for " + std::to_string(n)};
    for (const int64_t d : proj)
        if (d < 0 || d >= N)
            return std::pair{Error::wrong_value, name + "the coordinate " +
                                                     std::to_string(d) + " is not in [0, " +
                                                     std::to_string(N) + ")"};
    return std::nullopt;
}

} // namespace detail


// ===========================================  MAIN  =========================================== //

Set ContSet::project_high_dim(int64_t N, const std::vector<int64_t> &proj) const {
    detail::Entry entry;
    const detail::Refusal r = detail::project_high_dim_refusal(dim(), N, proj);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    return project_high_dim_impl(N, proj);
}

Set ContSet::project_high_dim_impl(int64_t N, const std::vector<int64_t> &proj) const {
    // I(:, proj): column j is the unit vector of coordinate proj[j]
    const int64_t n = static_cast<int64_t>(proj.size());
    std::vector<double> selection(N * n, 0.0);
    for (int64_t j = 0; j < n; ++j) selection[proj[j] * n + j] = 1.0;
    return linear_map_impl(Tensor::like(center_impl(), selection, {N, n}));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
