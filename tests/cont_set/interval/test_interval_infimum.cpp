// test_interval_infimum - interval infimum: the lower bound
//
// CORA: contSet/@interval/infimum.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void infimum_is_the_lower_bound(const std::string &b) {
    const Interval I(column({-1.0, 0.0, 2.0}), column({3.0, 0.5, 2.0}));
    check(close(I.infimum(), std::vector<double>{-1, 0, 2}, 0), b + ": the lower bound");
    check(close(I.infimum(), I.inf(), 0), b + ": inf");
    const Interval S = Interval::stack({I, I});
    check(S.infimum().shape() == std::vector<int64_t>({2, 3, 1}),
          b + ": a batch has a batch shape");
}

/// infimum has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0, 0.0}), column({1.0, 1.0}));
    const Tensor on = I.infimum();
    ChecksScope off(false);
    check(close(I.infimum(), on, 0), "infimum: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(infimum_is_the_lower_bound);
    checks_and_their_switch();
    return test::finish("interval infimum");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
