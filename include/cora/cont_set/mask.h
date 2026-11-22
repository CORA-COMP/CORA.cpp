// mask - what a predicate answers for a batch of sets: one flag per member (#163)
//
// A predicate on a batch (contains, is_intersecting, isequal, is_full_dim, is_bounded) says which
// members have the property, by a mask of the batch shape (0-d for one set). A Mask converts to
// bool as "every member does", so a call on one set, or a test of a whole batch, reads as before.
//
// Syntax:   Mask m = I.contains(p);   bool all = m;   Tensor flags = m.tensor();
// Outputs:  tensor - 1 for a member that has the property, else 0, of the batch shape
// See also: cont_set/empty_mask.h
//
// CORA.cpp only: CORA's predicates answer one logical for one set; the batch is CORA.cpp's.
// Authors:       Claude Opus 5.5, Tobias Ladner
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tensor.h"

#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Mask {
  public:
    /// The flags of the members: 0 or 1, of the batch shape.
    explicit Mask(Tensor flags) : flags_(std::move(flags)) {}

    /// The answer for one set, on the backend and dtype of `like`.
    Mask(bool answer, const Tensor &like) : flags_(like.full_like(Shape{}, answer ? 1.0 : 0.0)) {}

    /// The flags, 1 where the member has the property.
    const Tensor &tensor() const noexcept { return flags_; }

    /// Whether every member has the property (a batch without members: true).
    bool all() const { return flags_.numel() == 0 || flags_.min().item() != 0; }

    /// Whether some member has it.
    bool any() const { return flags_.numel() != 0 && flags_.max().item() != 0; }

    /// The answer for one set, "every member" for a batch.
    // NOLINTNEXTLINE(google-explicit-constructor): the answer for one set is its bool
    operator bool() const { return all(); }

  private:
    Tensor flags_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
