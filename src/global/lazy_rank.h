// lazy_rank - a ranking of keys produced front to back, for reductions that read its first entries
//
// The ranking is the stable order of the keys, the smaller first, ties in index order. It is the
// same order as a stable sort of the indices, but only the part that is asked for is sorted: a
// selection of a growing prefix, then a sort of that prefix. A ranking read to its first k
// entries costs O(m + k log k) instead of O(m log m). Keys are not NaN.
//
// Syntax:   LazyRank rank(keys);  while (!rank.empty()) { i = rank.peek().second; rank.pop(); }
// See also: reduce_adaptive.cpp, priv_poly_ops.cpp
//
// CORA.cpp only: the order is MATLAB's sort (stable); the laziness is CORA.cpp's.
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class LazyRank {
  public:
    using Entry = std::pair<double, std::size_t>; ///< the key and the index

    /// The ranking of the keys 0, 1, ...
    explicit LazyRank(const std::vector<double> &key) : entries_(key.size()) {
        for (std::size_t i = 0; i < key.size(); ++i) entries_[i] = {key[i], i};
    }

    /// The ranking of the entries (key, index); the indices must be increasing.
    explicit LazyRank(std::vector<Entry> &&entries) : entries_(std::move(entries)) {}

    /// Whether every entry has been popped.
    bool empty() const { return pos_ == entries_.size(); }

    /// The next entry (not empty); the entries up to it are sorted.
    const Entry &peek() {
        if (pos_ == sorted_) extend();
        return entries_[pos_];
    }

    void pop() { ++pos_; }

  private:
    static bool less(const Entry &a, const Entry &b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    }

    /// Sorts the next block: at least 256 entries, four times as many as are sorted already; the
    /// last block takes whatever remains once it is less than twice as long.
    void extend() {
        std::size_t end = std::max<std::size_t>(256, 4 * sorted_);
        if (2 * end >= entries_.size()) end = entries_.size();
        const auto from = entries_.begin() + static_cast<std::ptrdiff_t>(sorted_);
        const auto to = entries_.begin() + static_cast<std::ptrdiff_t>(end);
        if (end < entries_.size()) std::nth_element(from, to, entries_.end(), less);
        std::sort(from, to, less);
        sorted_ = end;
    }

    std::vector<Entry> entries_;
    std::size_t pos_ = 0, sorted_ = 0;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
