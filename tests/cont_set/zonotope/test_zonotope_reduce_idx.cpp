// test_zonotope_reduce_idx - zonotope reduce "idx": the chosen generators become their box
//
// CORA: contSet/@zonotope/private/priv_reduceIdx.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
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

/// The chosen generators are boxed and appended, the others keep their place and order.
void idx_boxes_the_chosen_generators(const std::string &b) {
    const Zonotope Z(Tensor({0.0, 0.0}), Tensor({{1.0, 2.0, -3.0}, {0.0, 0.0, 1.0}}));
    const Zonotope R = Z.reduce_idx({0, 2});
    check(test::close(R.generators(), Tensor({{2.0, 4.0, 0.0}, {0.0, 0.0, 1.0}}), 0),
          b + ": the kept generator, then the box |1| + |-3|, |0| + |1|");
    check(test::close(R.center(), Z.center(), 0), b + ": the center stays");
    check(test::close(Z.reduce_idx({}).generators(), Z.generators(), 0), b + ": no indices");
}

/// The reduction contains the zonotope: the support only grows along any direction.
void idx_reduction_encloses(const std::string &b) {
    cora::Rng rng(12);
    const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, 10);
    const Zonotope R = Z.reduce_idx({1, 4, 7});
    check(R.generators().shape() == std::vector<int64_t>({3, 10}), b + ": 7 kept and 3 boxed");
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = test::random_direction(rng, 3);
        check(test::support(R, d) >= test::support(Z, d) - 1e-9, b + ": it contains Z");
    }
}

/// An index past the generators is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(Tensor({0.0, 0.0}), Tensor({{1.0, 2.0}, {0.0, 1.0}}));
    check(test::raises([&] { Z.reduce_idx({2}); }, Error::wrong_value), "reduce_idx: index 2 of 2");
    check(test::raises([&] { Z.reduce_idx({-1}); }, Error::wrong_value), "reduce_idx: index -1");
    const Tensor on = Z.reduce_idx({0}).generators();
    ChecksScope off(false);
    check(test::close(Z.reduce_idx({0}).generators(), on, 0),
          "reduce_idx: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        idx_boxes_the_chosen_generators(b);
        idx_reduction_encloses(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope reduce idx");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
