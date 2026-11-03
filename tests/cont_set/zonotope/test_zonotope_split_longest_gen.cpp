// test_zonotope_split_longest_gen - zonotope split_longest_gen: the halves along the longest
// generator
//
// CORA: contSet/@zonotope/splitLongestGen.m
// PROPERTY: contSet/@zonotope/splitLongestGen.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// G = [(1, 0), (0.1, 0.1), (0, 3)]: the third is the longest; CORA's first half is at c + g/2.
void longest_generator_by_hand(const std::string &b) {
    const Zonotope Z(column({1.0, 1.0}), Tensor({{1.0, 0.1, 0.0}, {0.0, 0.1, 3.0}}));
    const auto [first, second] = Z.split_longest_gen();
    check(close(first.center(), std::vector<double>{1, 2.5}, 1e-14) &&
              close(second.center(), std::vector<double>{1, -0.5}, 1e-14),
          b + ": the centers c +- g/2, the plus first");
    check(close(first.generators(), std::vector<double>{1, 0.1, 0, 0, 0.1, 1.5}, 1e-14) &&
              close(second.generators(), first.generators(), 0),
          b + ": the longest generator is halved in both");
    const auto [a, c] = Zonotope(column({1.0}), Tensor::zeros({1, 0})).split_longest_gen();
    check(a.generators().shape().back() == 0 && c.generators().shape().back() == 0,
          b + ": no generator: the set twice");
    const auto [t, u] = Zonotope(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})).split_longest_gen();
    check(close(t.center(), std::vector<double>{0.5, 0}, 0), b + ": ties split the first");
}

/// Stated property: the halves cover Z, and are those of split along the generator of largest
/// length.
void halves_cover_the_set(const std::string &b) {
    cora::Rng rng(111);
    const Zonotope Z = Zonotope::generate_random(3, 5, rng);
    const auto [first, second] = Z.split_longest_gen();
    const Tensor length = Z.generator_length();
    const auto [lower, upper] = Z.split(int64_t(length.argmax(-1).item()), 100);
    check(close(second.center(), lower.center(), 1e-13) && close(first.center(), upper.center(), 1e-13),
          b + ": the longest generator");
    const Tensor points = Z.rand_point(40, rng);
    for (int k = 0; k < 40; ++k) {
        const Tensor p = points.slice(-1, k, k + 1);
        check(first.contains(p, "exact", 1e-9) || second.contains(p, "exact", 1e-9),
              b + ": a point of Z is in neither half");
    }
}

/// A batch is refused; with the checks off the halves are the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    check(test::raises([&] { Zonotope::stack({Z, Z}).split_longest_gen(); }, Error::not_supported),
          "split_longest_gen: a batch");
    const Tensor on = Z.split_longest_gen().first.center();
    ChecksScope off(false);
    check(close(Z.split_longest_gen().first.center(), on, 0), "split_longest_gen: checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        longest_generator_by_hand(b);
        halves_cover_the_set(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope split_longest_gen");
}
