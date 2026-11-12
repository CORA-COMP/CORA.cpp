// test_lazy_rank - LazyRank: the stable order of the keys, produced front to back
//
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "global/lazy_rank.h"
#include "testing.h"

#include <numeric>
#include <random>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The indices of the keys in the stable order, smallest first.
std::vector<std::size_t> stable_order(const std::vector<double> &key) {
    std::vector<std::size_t> idx(key.size());
    std::iota(idx.begin(), idx.end(), std::size_t(0));
    std::stable_sort(idx.begin(), idx.end(),
                     [&](std::size_t a, std::size_t b) { return key[a] < key[b]; });
    return idx;
}

/// The first `count` entries of the lazy ranking.
std::vector<std::size_t> first_entries(const std::vector<double> &key, std::size_t count) {
    LazyRank rank(key);
    std::vector<std::size_t> out;
    while (!rank.empty() && out.size() < count) {
        out.push_back(rank.peek().second);
        rank.pop();
    }
    return out;
}

/// Many ties, sizes around the block boundaries: the whole ranking and every prefix equal the
/// stable sort.
void matches_stable_sort() {
    std::mt19937 gen(7);
    for (const std::size_t m : {0, 1, 5, 255, 256, 257, 700, 1024, 1025, 5000}) {
        std::vector<double> key(m);
        for (double &k : key) k = double(gen() % 50);
        const std::vector<std::size_t> expected = stable_order(key);
        check(first_entries(key, m) == expected,
              "lazy rank: whole ranking of " + std::to_string(m));
        const std::size_t part = m / 3;
        check(
            first_entries(key, part) ==
                std::vector<std::size_t>(expected.begin(), expected.begin() + std::ptrdiff_t(part)),
            "lazy rank: prefix of " + std::to_string(m));
    }
}

/// Entries given with their indices keep those indices, and peek reports the key.
void entries_and_keys() {
    LazyRank rank(std::vector<LazyRank::Entry>{{2.0, 3}, {1.0, 5}, {2.0, 9}, {0.5, 11}});
    check(!rank.empty() && rank.peek().first == 0.5 && rank.peek().second == 11,
          "lazy rank: smallest");
    rank.pop();
    check(rank.peek().second == 5, "lazy rank: next");
    rank.pop();
    check(rank.peek().second == 3, "lazy rank: tie in index order");
    rank.pop();
    check(rank.peek().second == 9, "lazy rank: last");
    rank.pop();
    check(rank.empty(), "lazy rank: empty at the end");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    matches_stable_sort();
    entries_and_keys();
    return test::finish("lazy rank");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
