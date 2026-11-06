// priv_containment_vertex_enumeration - zonotope in zonotope, exactly, by a search over vertices
//
// The inner zonotope lies in the outer one iff all its vertices do, i.e. iff the outer norm of
// c_in - c_out + G_in nu is at most 1 for every nu in {-1, 1}^m. The search assigns the factors
// nu one at a time, longest generator first, and cuts a branch as soon as no completion can
// exceed 1 + tol.
//
// Syntax:   bool res = priv_containment_vertex_enumeration(inner, outer, tol);
// Inputs:   inner, outer - zonotopes
//           tol - allowed excess of the outer norm over 1
// Outputs:  res - true if inner is in outer (up to tol)
// See also: priv_zonotope_norm
//
// CORA: contSet/@zonotope/private/priv_zonotopeContainment_vertexEnumeration.m
// CORA authors: Adrian Kulmburg
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A node of the search: the factors assigned so far (0: free) and the outer norm it reaches.
struct Node {
    std::vector<int> nu;
    double value;
};

/// The outer norm of G nu for the columns of G (n x k, row-major).
double aux_norm_of(const HostZonotope &outer, const std::vector<double> &G, int64_t n, int64_t k,
                   const std::vector<int> &nu) {
    std::vector<double> point(n, 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < k; ++j) point[i] += G[i * k + j] * nu[j];
    return priv_zonotope_norm(outer, point);
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool priv_containment_vertex_enumeration(const HostZonotope &inner, const HostZonotope &outer,
                                         double tol) {
    // the columns (c_in - c_out, G_in), longest in the outer norm first
    const int64_t n = inner.n, k = inner.m + 1;
    std::vector<double> G(std::size_t(n * k));
    for (int64_t i = 0; i < n; ++i) {
        G[i * k] = inner.c[i] - outer.c[i];
        for (int64_t j = 0; j + 1 < k; ++j) G[i * k + 1 + j] = inner.G[i * inner.m + j];
    }
    std::vector<double> length(k);
    for (int64_t j = 0; j < k; ++j) {
        std::vector<double> column(n);
        for (int64_t i = 0; i < n; ++i) column[i] = G[i * k + j];
        length[j] = priv_zonotope_norm(outer, column);
    }
    std::vector<int64_t> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t a, int64_t b) { return length[a] > length[b]; });
    std::vector<double> sorted(std::size_t(n * k)), weight(k);
    for (int64_t j = 0; j < k; ++j) {
        weight[j] = length[order[j]];
        for (int64_t i = 0; i < n; ++i) sorted[i * k + j] = G[i * k + order[j]];
    }
    // what a node can still add: the lengths of the generators that are not at +-1 yet
    const auto heuristic = [&](const Node &node) {
        double h = node.value;
        for (int64_t j = 0; j < k; ++j) h += weight[j] * std::abs(std::abs(node.nu[j]) - 1);
        return h;
    };

    std::vector<Node> queue{{std::vector<int>(k, 0), 0.0}};
    while (!queue.empty()) {
        const Node current = queue.back();
        queue.pop_back();
        const auto free = std::find(current.nu.begin(), current.nu.end(), 0);
        if (free == current.nu.end()) continue;
        Node positive = current, negative = current;
        positive.nu[free - current.nu.begin()] = 1;
        negative.nu[free - current.nu.begin()] = -1;
        positive.value = aux_norm_of(outer, sorted, n, k, positive.nu);
        negative.value = aux_norm_of(outer, sorted, n, k, negative.nu);
        // the child with the larger norm is explored first
        const bool positive_first = positive.value <= negative.value;
        const Node &small = positive_first ? positive : negative;
        const Node &large = positive_first ? negative : positive;
        if (large.value > 1 + tol) return false;
        if (heuristic(small) <= 1 + tol) {
            if (heuristic(large) > 1 + tol) queue.push_back(large);
            continue;
        }
        // keep the queue ordered by value: small goes before the first node that is not smaller
        auto at = std::find_if(queue.begin(), queue.end(),
                               [&](const Node &node) { return node.value >= small.value; });
        queue.insert(at, small);
        queue.push_back(large);
    }
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
