// test_lean_zonotope_maps - origin, project_high_dim, representsa and the splits of
//
// Zonotope against the CORALean oracle
//
// ORACLE: contSet/@zonotope/origin.m
// ORACLE: contSet/@zonotope/projectHighDim_.m
// ORACLE: contSet/@zonotope/representsa_.m
// ORACLE: contSet/@zonotope/split.m
// ORACLE: contSet/@zonotope/splitFirstGen.m
// ORACLE: contSet/@zonotope/splitLongestGen.m
//
// The oracle splits along a generator of Z (CORA.cpp: split(N, max_order) with no reduction) and
// returns the halves c - g/2 first; CORA's splitLongestGen lists c + g/2 first, so its halves
// are compared swapped.
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;


namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The result of a zonotope-valued operation against the oracle's: the same bits at nearest; in
/// the sound mode inside the enclosure, by the interval hulls.
template <class A, class B>
void aux_check_set(const A &mine, const B &theirs) {
    if (oracle::mode() == oracle::Mode::exact) return oracle::check(mine, theirs);
    oracle::check(mine.interval(), theirs.enclosure().interval());
}

/// The two halves of a split: the same bits at nearest; in the sound mode each inside the
/// oracle's by the interval hulls (the oracle's containment proof is too slow for the splits).
template <class A, class B>
void aux_check_halves(const A &mine, const B &theirs) {
    if (oracle::mode() == oracle::Mode::exact) return oracle::check(mine, theirs);
    aux_check_set(mine.first, theirs.first);
    aux_check_set(mine.second, theirs.second);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const std::vector<std::pair<int, int>> sizes{{2, 3}, {5, 30}, {20, 50}};

    // up to 5 dimensions: the oracle's enclosure of the 20-dimensional origin does not return
    oracle::run("zonotope.origin", [&](oracle::Random &) {
        for (const int n : {1, 2, 5}) oracle::check(Zonotope::origin(n), lean::Zonotope::origin(n));
    });

    oracle::run("zonotope.projectHighDim", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            std::vector<int64_t> dims(n); // the states reversed into the last n of n + 3
            for (int i = 0; i < n; ++i) dims[i] = n + 2 - i;
            aux_check_set(Z.project_high_dim(n + 3, dims),
                          lean::Zonotope(Z).project_high_dim(n + 3, dims));
        }
    });

    oracle::run("zonotope.representsa", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Tensor zeros = Tensor::zeros({n, m}), c = rnd.vector(n);
            const Zonotope random = rnd.zonotope(n, m), point(c, zeros),
                           origin(c.zeros_like(), zeros);
            for (const char *type : {"point", "origin"})
                for (const Zonotope &Z : {random, point, origin})
                    oracle::check(Z.representsa(type, 0.0), lean::Zonotope(Z).representsa(type));
        }
    });

    oracle::run("zonotope.split", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            for (const int64_t gen : {int64_t(0), int64_t(m - 1)})
                aux_check_halves(Z.split(gen, 1000), lean::Zonotope(Z).split(gen));
        }
    });

    oracle::run("zonotope.splitFirstGen", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            aux_check_halves(Z.split_first_gen(), lean::Zonotope(Z).split_first_gen());
        }
    });

    oracle::run("zonotope.splitLongestGen", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            const auto mine = Z.split_longest_gen();
            aux_check_halves(std::pair{mine.second, mine.first},
                             lean::Zonotope(Z).split_longest_gen());
        }
    });
    return test::finish("lean zonotope maps");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
