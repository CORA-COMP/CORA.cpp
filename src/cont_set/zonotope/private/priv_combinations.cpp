// priv_combinations - the n-subsets of {0, ..., total - 1}, and the matrices they pick from G
//
// Syntax:   Combinations subsets(total, n);   flat = subsets.next(count);
//           Tensor P = priv_column_subsets(G, members);   Tensor d = priv_abs_det(P);
// Inputs:   total, n - the subsets of size n of total indices
//           count - how many to produce;
//           G - (..., r, m) columns;  members - (B, n) column indices, one subset per row
// Outputs:  next - up to `count` subsets in lexicographic order as one flat list (empty when
//           done);  P - (..., B, r, n), the columns G(:, members(b, :));  d - |det| of the
//           matrices (..., n, n), as (..., 1, 1)
// See also: dominant_directions, volume
//
// CORA.cpp only: CORA's batchCombinator and its determinants, shared by volume and
//                dominant_directions
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Combinations::Combinations(int64_t total, int64_t n) : total_(total), n_(n), current_(n) {
    // the first subset is {0, ..., n - 1}
    for (int64_t i = 0; i < n; ++i) current_[i] = i;
    done_ = n > total || n < 1;
}

std::vector<int64_t> Combinations::next(int64_t count) {
    std::vector<int64_t> flat;
    for (int64_t produced = 0; produced < count && !done_; ++produced) {
        flat.insert(flat.end(), current_.begin(), current_.end());
        // the successor: raise the last index that has room, and restart those after it
        int64_t i = n_ - 1;
        while (i >= 0 && current_[i] == total_ - n_ + i) --i;
        if (i < 0) {
            done_ = true;
            break;
        }
        ++current_[i];
        for (int64_t j = i + 1; j < n_; ++j) current_[j] = current_[j - 1] + 1;
    }
    return flat;
}

Tensor priv_column_subsets(const Tensor &G, const Tensor &members) {
    const int64_t B = members.shape()[0], n = members.shape()[1], batch = G.ndim() - 2;
    // the shape (..., r, B, n) of the picked columns, grouped by subset
    Shape shape;
    for (int64_t i = 0; i < batch; ++i) shape.push_back(G.shape()[i]);
    shape.push_back(G.shape()[batch]);
    shape.push_back(B);
    shape.push_back(n);
    // (..., r, B * n) -> (..., r, B, n) -> (..., B, r, n): exchange the axes r and B
    std::vector<int> axes(batch + 3);
    for (int64_t i = 0; i < batch; ++i) axes[i] = int(i);
    axes[batch] = int(batch) + 1;
    axes[batch + 1] = int(batch);
    axes[batch + 2] = int(batch) + 2;
    return G.take(members.reshape({B * n}), -1).reshape(shape).permute_dims(axes);
}

Tensor priv_abs_det(const Tensor &P) { return P.qr().second.diagonal().prod(-2, true); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
