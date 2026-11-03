// lift - a box placed into a higher-dimensional space, as CORA's interval.lift_
//
// Syntax:   Interval R = I.lift(N, dims);
// Inputs:   N - dimension of the space, N >= dim;  dims - the N-dimensional coordinates (0-based)
//           of the dim coordinates of I, in the order of I
// Outputs:  R - the box with coordinate dims[j] equal to that of I and (-Inf, Inf) elsewhere; for
//           a repeated entry of dims the last one counts, as CORA's assignment
// See also: project
//
// CORA: contSet/@interval/lift_.m

#include "cont_set/interval/private/priv_embed.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    CORA_CHECK(N >= dim(), Error::wrong_value, [&] {
        return "lift: the space of " + std::to_string(N) +
               " dimensions must not be smaller than the interval of " + std::to_string(dim());
    });
    CORA_CHECK(static_cast<int64_t>(dims.size()) == dim(), Error::wrong_value, [&] {
        return "lift: one coordinate per dimension of the interval is needed, got " +
               std::to_string(dims.size()) + " for " + std::to_string(dim());
    });
    for (const int64_t d : dims)
        CORA_CHECK(d >= 0 && d < N, Error::wrong_value, [&] {
            return "lift: the coordinate " + std::to_string(d) + " is not in [0, " +
                   std::to_string(N) + ")";
        });
    CORA_CHECK(!is_empty(), Error::not_supported, [] {
        return std::string("lift: the operation is not supported for empty intervals");
    });
    const double kInf = std::numeric_limits<double>::infinity();
    return priv_embed(*this, N, dims, -kInf, kInf);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
