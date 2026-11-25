// priv_split - a simple boundary loop as convex pieces: ear clipping, merged greedily
//
// The loop is turned counter-clockwise; a convex loop is its own piece. Otherwise an ear (a left
// turn with no other vertex in its triangle) is cut off until a triangle is left, and two pieces
// that share an edge are merged while the merge stays convex (Hertel and Mehlhorn's greedy step),
// as CORA's splitIntoConvexSets merges the triangles of the polyshape.
//
// Syntax:   std::vector<Piece> pieces = priv_convex_split(loop);
// See also: priv.h, polygon.cpp
//
// CORA.cpp only: the split of a boundary that MATLAB's polyshape keeps as it is
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"

#include <algorithm>
#include <cstddef>
#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

double aux_cross(Point a, Point b, Point c) {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

/// Whether the counter-clockwise loop turns right nowhere.
bool aux_convex(const Piece &loop) {
    const std::size_t n = loop.size();
    for (std::size_t i = 0; i < n; ++i)
        if (aux_cross(loop[i], loop[(i + 1) % n], loop[(i + 2) % n]) < 0) return false;
    return true;
}

/// Whether x lies in the closed counter-clockwise triangle a, b, c.
bool aux_in_triangle(Point a, Point b, Point c, Point x) {
    return aux_cross(a, b, x) >= 0 && aux_cross(b, c, x) >= 0 && aux_cross(c, a, x) >= 0;
}

/// The triangles of the counter-clockwise simple loop, ear by ear.
std::vector<Piece> aux_ears(Piece loop) {
    std::vector<Piece> triangles;
    while (loop.size() > 3) {
        const std::size_t n = loop.size();
        std::size_t ear = n;
        for (std::size_t i = 0; i < n && ear == n; ++i) {
            const Point a = loop[(i + n - 1) % n], b = loop[i], c = loop[(i + 1) % n];
            if (aux_cross(a, b, c) <= 0) continue;
            // no other vertex in the triangle
            bool empty = true;
            for (std::size_t j = 0; j < n && empty; ++j)
                if (loop[j] != a && loop[j] != b && loop[j] != c)
                    empty = !aux_in_triangle(a, b, c, loop[j]);
            if (empty) ear = i;
        }
        // a loop without an ear is not simple: its rest is dropped
        if (ear == n) return triangles;
        triangles.push_back({loop[(ear + n - 1) % n], loop[ear], loop[(ear + 1) % n]});
        loop.erase(loop.begin() + std::ptrdiff_t(ear));
    }
    if (aux_cross(loop[0], loop[1], loop[2]) > 0) triangles.push_back(loop);
    return triangles;
}

/// A and B joined along an edge they share (reversed), if the result is convex.
std::optional<Piece> aux_merge(const Piece &A, const Piece &B) {
    const std::size_t na = A.size(), nb = B.size();
    for (std::size_t i = 0; i < na; ++i)
        for (std::size_t j = 0; j < nb; ++j) {
            if (A[i] != B[(j + 1) % nb] || A[(i + 1) % na] != B[j]) continue;
            // A from i + 1 around to i, then B after A[i] up to the vertex before A[i + 1]
            Piece merged;
            for (std::size_t k = 1; k <= na; ++k) merged.push_back(A[(i + k) % na]);
            for (std::size_t k = 2; k < nb; ++k) merged.push_back(B[(j + k) % nb]);
            if (aux_convex(merged)) return merged;
        }
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Piece> priv_convex_split(Piece loop) {
    // no repeated neighbors, and no closing copy of the first vertex
    loop.erase(std::unique(loop.begin(), loop.end()), loop.end());
    while (loop.size() > 1 && loop.front() == loop.back()) loop.pop_back();
    if (loop.size() < 3) return loop.empty() ? std::vector<Piece>{} : std::vector<Piece>{loop};
    if (priv_area(loop) < 0) std::reverse(loop.begin(), loop.end());
    if (aux_convex(loop)) return {loop};
    std::vector<Piece> pieces = aux_ears(std::move(loop));
    // merge two pieces while one merge stays convex
    for (bool merged = true; merged;) {
        merged = false;
        for (std::size_t a = 0; a < pieces.size() && !merged; ++a)
            for (std::size_t b = a + 1; b < pieces.size() && !merged; ++b)
                if (std::optional<Piece> m = aux_merge(pieces[a], pieces[b])) {
                    pieces[a] = std::move(*m);
                    pieces.erase(pieces.begin() + std::ptrdiff_t(b));
                    merged = true;
                }
    }
    return pieces;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
