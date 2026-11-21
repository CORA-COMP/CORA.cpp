// test_con_zonotope_is_bounded - ConZonotope::is_bounded: always true
//
// CORA: contSet/@conZonotope/isBounded.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const ConZonotope cz(vec({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 3.0}}),
                             Tensor({{1.0, 1.0}}), vec({0.5}));
        check(cz.is_bounded(), b + ": a constrained zonotope is bounded");
        check(ConZonotope::empty(2).is_bounded(), b + ": the empty set is bounded");
        check(Set(cz)->is_bounded(), b + ": through the set interface");
        const ConZonotope batch(Tensor::zeros({2, 2}), Tensor::ones({2, 2, 3}).matrix_transpose());
        check(batch.is_bounded(), b + ": a batch is bounded");
    });
    // the question takes no input to get wrong; with the checks off the answer is the same
    ChecksScope off(false);
    check(ConZonotope::empty(1).is_bounded(), "is_bounded: the same with the checks off");
    return test::finish("con_zonotope is_bounded");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
