// dimension_set - the base of the sets given by their dimension alone (EmptySet, Fullspace)
//
// It holds the dimension n, answers dim(), checks n >= 0 at construction and tells why a lift
// is refused; the derived class gives everything else.
//
// Syntax:   class EmptySet : public detail::DimensionSet { ... };
// See also: cont_set/empty_set/empty_set.h, cont_set/fullspace/fullspace.h
//
// CORA.cpp only: CORA's emptySet and fullspace each store their dimension themselves.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/global/error.h"

#include <cstdint>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::detail {

class DimensionSet : public ContSet {
  protected:
    /// A set of class id in n >= 0 dimensions (Error::wrong_input_in_constructor otherwise).
    DimensionSet(ClassId id, int64_t n) : ContSet(id), n_(n) {
        detail::Entry entry;
        CORA_CHECK(n >= 0, Error::wrong_input_in_constructor, [&] {
            return std::string(::cora::class_name(id)) +
                   ": the dimension must be a nonnegative integer, got " + std::to_string(n);
        });
    }

    /// Why lift(N, dims) is refused ("" if it is not): it needs N >= n and one position per
    /// dimension, each in [0, N). The derived lift checks it (Error::wrong_value); `operation`
    /// names it in the message (projectHighDim_ has the same checks).
    std::string lift_refusal(int64_t N, const std::vector<int64_t> &dims,
                             const char *operation = "lift") const;

    int64_t dim_impl() const override { return n_; }

    int64_t n_;
};

} // namespace cora::detail

// ---------------------------------------  END OF CODE  ---------------------------------------- //
