// empty_mask - the flags of the empty members of a batch of sets (#163)
//
// Syntax:   const EmptyMask &m = S.empty_mask();   EmptyMask m(batch, flags);   m.flags(batch);
// Inputs:   batch - the batch shape of the set
//           flags - one flag per member, row major (nonzero: empty)
// Outputs:  the mask, or the flags broadcast to a batch shape as numpy broadcasts
// See also: ContSet::empty_mask, Interval::empty
//
// CORA.cpp only: CORA stores an empty set in its data (zeros(n, 0), an infeasible constraint).
// Authors:       Claude Opus 5.5, Tobias Ladner
// Written:       19-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/empty_mask.h"
#include "cora/global/error.h"
#include "cora/global/settings.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The flags of `shape` broadcast to `batch` (numpy's rule, axes aligned at the end).
std::vector<uint8_t> aux_broadcast(const Shape &shape, const std::vector<uint8_t> &flags,
                                   const Shape &batch) {
    const int64_t rank = batch.size(), offset = rank - shape.size();
    if (offset < 0)
        CORA_FAIL(Error::dimension_mismatch, "EmptyMask: the batch " + shape_text(shape) +
                                                 " does not broadcast to " + shape_text(batch));
    for (int64_t a = 0; a < shape.size(); ++a)
        if (shape[a] != 1 && shape[a] != batch[a + offset])
            CORA_FAIL(Error::dimension_mismatch, "EmptyMask: the batch " + shape_text(shape) +
                                                     " does not broadcast to " + shape_text(batch));
    std::vector<uint8_t> out(batch.numel());
    std::vector<int64_t> index(rank, 0);
    for (uint8_t &flag : out) {
        int64_t source = 0;
        for (int64_t a = 0; a < shape.size(); ++a)
            source = source * shape[a] + (shape[a] == 1 ? 0 : index[a + offset]);
        flag = flags[source];
        for (int64_t a = rank - 1; a >= 0; --a) // the next index, row major
            if (++index[a] < batch[a])
                break;
            else
                index[a] = 0;
    }
    return out;
}

/// The batch shape both shapes broadcast to.
Shape aux_joint(const Shape &a, const Shape &b) {
    const int64_t rank = std::max(a.size(), b.size());
    Shape out;
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t x = i - (rank - a.size()) >= 0 ? a[i - (rank - a.size())] : 1;
        const int64_t y = i - (rank - b.size()) >= 0 ? b[i - (rank - b.size())] : 1;
        if (x != y && x != 1 && y != 1)
            CORA_FAIL(Error::dimension_mismatch, "EmptyMask: the batches " + shape_text(a) +
                                                     " and " + shape_text(b) + " do not broadcast");
        out.push_back(x == 1 ? y : x);
    }
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Construction ------------------------------------------------------------------------------------

EmptyMask::EmptyMask(const Shape &batch, std::vector<uint8_t> flags) {
    detail::Entry entry;
    if (static_cast<int64_t>(flags.size()) != batch.numel())
        CORA_FAIL(Error::dimension_mismatch, "EmptyMask: " + std::to_string(flags.size()) +
                                                 " flags for the batch " + shape_text(batch));
    const auto set = std::count_if(flags.begin(), flags.end(), [](uint8_t f) { return f != 0; });
    if (set == 0) return; // no member is empty
    if (set == static_cast<int64_t>(flags.size())) {
        state_ = State::all;
        return;
    }
    // some members are empty: keep the flags, normalized to 0 and 1
    state_ = State::mixed;
    shape_ = batch;
    for (uint8_t &f : flags) f = f != 0;
    flags_ = std::move(flags);
}

// Queries -----------------------------------------------------------------------------------------

std::vector<uint8_t> EmptyMask::flags(const Shape &batch) const {
    detail::Entry entry;
    if (state_ != State::mixed) return std::vector<uint8_t>(batch.numel(), state_ == State::all);
    return aux_broadcast(shape_, flags_, batch);
}

// Combination -------------------------------------------------------------------------------------

EmptyMask EmptyMask::either(const EmptyMask &other) const {
    detail::Entry entry;
    // a uniform mask decides alone; else the flags of the joint batch, or-ed
    if (state_ == State::all || other.state_ == State::none) return *this;
    if (other.state_ == State::all || state_ == State::none) return other;
    const Shape batch = aux_joint(shape_, other.shape_);
    std::vector<uint8_t> a = flags(batch);
    const std::vector<uint8_t> b = other.flags(batch);
    for (std::size_t k = 0; k < a.size(); ++k) a[k] = a[k] | b[k];
    return {batch, std::move(a)};
}

EmptyMask EmptyMask::both(const EmptyMask &other) const {
    detail::Entry entry;
    // a uniform mask decides alone; else the flags of the joint batch, and-ed
    if (state_ == State::none || other.state_ == State::all) return *this;
    if (other.state_ == State::none || state_ == State::all) return other;
    const Shape batch = aux_joint(shape_, other.shape_);
    std::vector<uint8_t> a = flags(batch);
    const std::vector<uint8_t> b = other.flags(batch);
    for (std::size_t k = 0; k < a.size(); ++k) a[k] = a[k] & b[k];
    return {batch, std::move(a)};
}

EmptyMask EmptyMask::stack(const std::vector<EmptyMask> &masks, const Shape &batch) {
    detail::Entry entry;
    Shape stacked;
    stacked.push_back(static_cast<int64_t>(masks.size()));
    for (const int64_t d : batch) stacked.push_back(d);
    // the members' flags one after the other, each broadcast to the batch
    std::vector<uint8_t> flags;
    flags.reserve(stacked.numel());
    for (const EmptyMask &m : masks) {
        const std::vector<uint8_t> f = m.flags(batch);
        flags.insert(flags.end(), f.begin(), f.end());
    }
    return {stacked, std::move(flags)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
