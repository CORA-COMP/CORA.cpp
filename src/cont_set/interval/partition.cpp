// partition - a box cut into equal boxes, as CORA's interval.partition
//
// The first coordinate is the slowest. Piece k of a coordinate runs from inf + r(k) w to inf +
// r(k+1) w for the width w = sup - inf, where the fraction r(k) is the smaller (larger) of
// k/splits and 1 - (splits-k)/splits for a lower (upper) bound, as CORALean's partition. The
// neighbours then overlap or touch, so the pieces cover the box in any rounding.
//
// Syntax:   std::vector<Interval> P = I.partition(splits);
// Inputs:   splits - the number of pieces per coordinate, at least 1
// Outputs:  P - the splits^n boxes, in the order of nested loops over the coordinates
// See also: split, enclose_points
//
// CORA: contSet/@interval/partition.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The fractions of the width at the cuts 0..splits: the smaller and the larger of k/splits and
/// 1 - (splits-k)/splits.
std::pair<std::vector<double>, std::vector<double>> aux_fractions(int64_t splits) {
    std::vector<double> smaller, larger;
    for (int64_t k = 0; k <= splits; ++k) {
        const double up = double(k) / double(splits);
        const double down = 1.0 - double(splits - k) / double(splits);
        smaller.push_back(std::fmin(up, down));
        larger.push_back(std::fmax(up, down));
    }
    return {smaller, larger};
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Interval> Interval::partition(int64_t splits) const {
    detail::Entry entry;
    CORA_CHECK(splits >= 1, Error::wrong_value, [&] {
        return "partition: the number of splits must be positive, got " + std::to_string(splits);
    });
    CORA_CHECK(!is_empty() && inf_.size(-1) == 1, Error::not_supported, [] {
        return std::string("partition: only a nonempty box (not a matrix) can be partitioned");
    });
    const int64_t n = dim();
    int64_t count = 1;
    for (int64_t i = 0; i < n; ++i) {
        CORA_CHECK(count <= numerics::kMaxPieces / splits, Error::out_of_memory, [&] {
            return "partition: " + std::to_string(splits) + "^" + std::to_string(n) +
                   " boxes are more than " + std::to_string(numerics::kMaxPieces);
        });
        count *= splits;
    }
    const auto [smaller, larger] = aux_fractions(splits);
    const Tensor width = sup_ - inf_;
    std::vector<Interval> pieces;
    // one cut counter per coordinate
    std::vector<int64_t> cut(n, 0);
    for (int64_t piece = 0; piece < count; ++piece) {
        // the cuts of the piece, the first coordinate counting slowest
        std::vector<double> from(n), to(n);
        for (int64_t i = 0, rest = piece; i < n; ++i) {
            const int64_t j = n - 1 - i;
            cut[j] = rest % splits;
            rest /= splits;
        }
        for (int64_t i = 0; i < n; ++i) {
            from[i] = smaller[cut[i]];
            to[i] = larger[cut[i] + 1];
        }
        pieces.emplace_back(inf_ + width.multiply(Tensor::like(inf_, from, {n, 1})),
                            inf_ + width.multiply(Tensor::like(inf_, to, {n, 1})));
    }
    return pieces;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
