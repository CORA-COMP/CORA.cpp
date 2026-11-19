// empty_mask - which members of a batch of sets are the empty set (#163)
//
// A set that can be empty says so by a mask of its batch shape, one flag per member, and never by
// its data: no zero columns, negative radii or infeasible rows. The data of an empty member keeps
// the shapes of the set (n stays) and is not read. The mask lives in ContSet, so every class
// adopts it the same way; a set without empty members holds no flags at all.
//
// Syntax:   EmptyMask none;   EmptyMask all = EmptyMask::all();   bool e = S.empty_mask().all();
// See also: cont_set/cont_set.h, cont_set/interval/interval.h
//
// CORA.cpp only: CORA stores an empty set in its data (zeros(n, 0), an infeasible constraint).
// Authors:       Claude Opus 5.5, Tobias Ladner
// Written:       19-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tensor.h"

#include <cstdint>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class EmptyMask {
  public:
    /// No member is empty.
    EmptyMask() = default;

    /// Every member is empty, whatever the batch shape.
    static EmptyMask all_members() {
        EmptyMask m;
        m.state_ = State::all;
        return m;
    }

    /// One flag per member of the batch shape `batch`, row major (nonzero: empty).
    EmptyMask(const Shape &batch, std::vector<uint8_t> flags);

    /// Whether some member, or every member, is empty (a batch without members has none).
    bool any() const noexcept { return state_ != State::none; }
    bool all() const noexcept { return state_ == State::all; }

    /// The flags of the members of `batch`, row major: the mask broadcast to it.
    std::vector<uint8_t> flags(const Shape &batch) const;

    /// The mask of a result that is empty where either operand is (a sum, a product), or where
    /// both are (a hull, a union).
    EmptyMask either(const EmptyMask &other) const;
    EmptyMask both(const EmptyMask &other) const;

    /// The masks of sets of one batch shape `batch`, stacked along a new first batch axis.
    static EmptyMask stack(const std::vector<EmptyMask> &masks, const Shape &batch);

    friend bool operator==(const EmptyMask &a, const EmptyMask &b) {
        return a.state_ == b.state_ && a.shape_ == b.shape_ && a.flags_ == b.flags_;
    }

  private:
    enum class State : uint8_t { none, all, mixed };
    State state_ = State::none;
    Shape shape_;                // the batch shape of the flags (mixed only)
    std::vector<uint8_t> flags_; // one per member (mixed only)
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
