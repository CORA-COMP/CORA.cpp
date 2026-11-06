// test_zonotope_split_first_gen - zonotope split_first_gen: the halves along the generator with
//
// the largest score
//
// CORA: contSet/@zonotope/splitFirstGen.m
// PROPERTY: contSet/@zonotope/splitFirstGen.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// G = [(1, 0), (0.1, 0.1), (0, 3)]: the scores sum_k |g_j' g_k| are 1.1, 0.42 and 9.3, so
/// the third is halved: centers c -+ (0, 1.5) and that generator (0, 1.5).
void first_generator_by_hand(const std::string &b) {
    const Zonotope Z(column({1.0, 1.0}), Tensor({{1.0, 0.1, 0.0}, {0.0, 0.1, 3.0}}));
    const auto [lower, upper] = Z.split_first_gen();
    check(close(lower.center(), std::vector<double>{1, -0.5}, 1e-14) &&
              close(upper.center(), std::vector<double>{1, 2.5}, 1e-14),
          b + ": the centers c -+ g/2");
    check(close(upper.generators(), std::vector<double>{1, 0.1, 0, 0, 0.1, 1.5}, 1e-14),
          b + ": the third generator is halved");
    const auto [a, c] = Zonotope(column({1.0}), Tensor::zeros({1, 0})).split_first_gen();
    check(a.generators().shape().back() == 0 && c.generators().shape().back() == 0,
          b + ": no generator: the set twice");
    // equal scores: the first generator
    const auto [t, u] =
        Zonotope(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})).split_first_gen();
    check(close(t.center(), std::vector<double>{-0.5, 0}, 0), b + ": ties split the first");
}

/// Stated property: the halves cover Z, and are those of split along the generator of largest
/// score (found by the definition).
void halves_cover_the_set(const std::string &b) {
    cora::Rng rng(101);
    const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, 5);
    const auto [lower, upper] = Z.split_first_gen();
    const std::vector<double> G = Z.generators().data();
    int best = 0;
    double best_score = -1;
    for (int j = 0; j < 5; ++j) {
        double score = 0;
        for (int k = 0; k < 5; ++k) {
            double dot = 0;
            for (int i = 0; i < 3; ++i) dot += G[i * 5 + j] * G[i * 5 + k];
            score += std::abs(dot);
        }
        if (score > best_score) best_score = score, best = j;
    }
    const auto [l2, u2] = Z.split(best, 100);
    check(close(lower.center(), l2.center(), 1e-13) &&
              close(upper.generators(), u2.generators(), 1e-13),
          b + ": the generator of the largest score");
    // the pieces cover the points
    const Tensor points = Z.rand_point(40, rng);
    for (int k = 0; k < 40; ++k) {
        const Tensor p = points.slice(-1, k, k + 1);
        check(lower.contains(p, "exact", 1e-9) || upper.contains(p, "exact", 1e-9),
              b + ": a point of Z is in neither half");
    }
}

/// A batch is refused; with the checks off the halves are the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    check(test::raises([&] { Zonotope::stack({Z, Z}).split_first_gen(); }, Error::not_supported),
          "split_first_gen: a batch");
    const Tensor on = Z.split_first_gen().first.center();
    ChecksScope off(false);
    check(close(Z.split_first_gen().first.center(), on, 0), "split_first_gen: checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        first_generator_by_hand(b);
        halves_cover_the_set(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope split_first_gen");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
