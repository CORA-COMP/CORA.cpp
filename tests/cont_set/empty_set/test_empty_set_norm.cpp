// test_empty_set_norm - the norm of the empty set: -Inf, as CORA's contSet.norm returns it
//
// CORA: contSet/@contSet/norm.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The same -Inf for every type and mode, and for the empty box; a batch-free (1, 1) tensor.
void the_norm_is_minus_infinity(const std::string &b) {
    const std::vector<double> minus_inf{-std::numeric_limits<double>::infinity()};
    const EmptySet E(2);
    const ContSet &S = E;
    for (const Tensor::Norm type :
         {Tensor::Norm::one, Tensor::Norm::two, Tensor::Norm::inf, Tensor::Norm::fro})
        check(S.norm(type).data() == minus_inf && S.norm(type, "exact").shape() == Shape({1, 1}),
              b + ": the empty set");
    check(static_cast<const ContSet &>(Interval::empty(2)).norm().data() == minus_inf,
          b + ": the empty box");
}

/// A mode outside CORA's three is refused; with the checks off the call goes through.
void checks_and_their_switch() {
    const EmptySet E(2);
    check(test::raises([&] { static_cast<const ContSet &>(E).norm(Tensor::Norm::two, "bogus"); },
                       Error::wrong_value),
          "a wrong mode");
    ChecksScope off(false);
    check(static_cast<const ContSet &>(E).norm().shape() == Shape({1, 1}), "the checks off: shape");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_norm_is_minus_infinity);
    checks_and_their_switch();
    return test::finish("empty_set norm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
