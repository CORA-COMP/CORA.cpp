// test_zonotope_reduce_adaptive - zonotope reduce_adaptive: the generators of least metric boxed
//
// The generators are ranked by Girard's metric; those of metric 0 and as many more as stay within
// the share of the box are replaced by one box generator per dimension, the others stay in order.
// Hand-computed cases for the three regimes (nothing, one generator, everything), the box and
// the support (the result encloses the zonotope), and the arguments.
//
// CORA: contSet/@zonotope/reduce.m
// CORA: contSet/@zonotope/private/priv_reduceAdaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// metrics 1, 0.4 and 0.01; the box is (1.8, 1.41), its norm 2.2864
Zonotope aux_three() {
    return Zonotope(Tensor({1.0, 2.0}), Tensor({{1.0, 0.5, 0.3}, {1.0, -0.4, 0.01}}));
}

void the_share_decides_how_many_are_boxed(const std::string &b) {
    // the three regimes of the share, by hand
    const Zonotope Z = aux_three();
    // nothing to box: the three stay and two zero box generators follow
    const Zonotope none = Z.reduce_adaptive(0.0);
    check(none.generators().shape() == std::vector<int64_t>({2, 5}), b + ": no share, 5 columns");
    check(test::close(none.generators().slice(-1, 0, 3), Z.generators()), b + ": the same three");
    check(test::close(none.generators().slice(-1, 3, 5), Tensor::zeros({2, 2})),
          b + ": and no box");
    // the third (distance bound 0.02 <= 4.57 * 0.05): boxed, the first two stay in order
    const Zonotope one = Z.reduce_adaptive(0.05);
    check(one.generators().shape() == std::vector<int64_t>({2, 4}),
          b + ": a small share, 4 columns");
    check(test::close(one.generators(), Tensor({{1.0, 0.5, 0.3, 0.0}, {1.0, -0.4, 0.0, 0.01}})),
          b + ": the third is boxed");
    // everything: the box of Z
    const Zonotope all = Z.reduce_adaptive(1.0);
    check(test::close(all.generators(), Tensor({{1.8, 0.0}, {0.0, 1.41}})),
          b + ": a share of 1 leaves the box");
    check(test::close(all.center(), Z.center()), b + ": the center stays");
}

void the_result_encloses_the_zonotope(const std::string &b) {
    cora::Rng rng(9);
    const int n = 3;
    const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, 12);
    // every share from nothing to everything boxed
    for (const double share : {0.0, 0.02, 0.2, 1.0}) {
        const Zonotope R = Z.reduce_adaptive(share);
        check(R.generators().shape().back() <= 12 + n, b + ": at most h + n generators");
        // the box does not change, the support only grows
        check(test::close(R.interval().inf(), Z.interval().inf(), 1e-9) &&
                  test::close(R.interval().sup(), Z.interval().sup(), 1e-9),
              b + ": the box stays, share " + std::to_string(share));
        // the support function along random directions
        for (int k = 0; k < 30; ++k) {
            const std::vector<double> d = test::random_direction(rng, n);
            check(test::support(R, d) >= test::support(Z, d) - 1e-9,
                  b + ": the support only grows");
        }
    }
}

void a_zonotope_without_generators_gets_a_zero_box(const std::string &b) {
    const Zonotope Z(Tensor({1.0, 2.0}), Tensor::zeros({2, 0}));
    const Zonotope R = Z.reduce_adaptive(0.1);
    check(R.generators().shape() == std::vector<int64_t>({2, 2}), b + ": n box generators");
    check(test::close(R.generators(), Tensor::zeros({2, 2})), b + ": all zero");
}

void wrong_arguments_are_described(const std::string &b) {
    const Zonotope Z = aux_three();
    check(test::raises([&] { Z.reduce_adaptive(-0.1); }, Error::wrong_value), b + ": negative");
    check(test::raises([&] { Z.reduce_adaptive(std::nan("")); }, Error::wrong_value), b + ": NaN");
}

/// With the checks off the reduction is the same.
void checks_and_their_switch() {
    const Zonotope Z = aux_three();
    const Tensor on = Z.reduce_adaptive(0.05).generators();
    ChecksScope off(false);
    check(test::close(Z.reduce_adaptive(0.05).generators(), on, 0),
          "reduce_adaptive: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_share_decides_how_many_are_boxed(b);
        the_result_encloses_the_zonotope(b);
        a_zonotope_without_generators_gets_a_zero_box(b);
        wrong_arguments_are_described(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope reduce_adaptive");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
