// test_empty_set_support_function - EmptySet support_function: -Inf, the supremum over no point
//
// CORA: contSet/@emptySet/supportFunc_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;

namespace {

const double kInf = std::numeric_limits<double>::infinity();

void the_support_function_is_minus_infinity(const std::string &b) {
    const EmptySet E(2);
    check(test::support(E, {1.0, 2.0}) == -kInf, b + ": -Inf");
    const Tensor batch = Tensor::from_data({1, 0, 0, 1, 1, 1}, {3, 2, 1});
    const Tensor rho = E.support_function(batch);
    check(rho.shape() == std::vector<int64_t>({3, 1, 1}) && rho.min().item() == -kInf &&
              rho.max().item() == -kInf,
          b + ": a batch of three directions");
    check(EmptySet(0).support_function(Tensor::zeros({0, 1})).data()[0] == -kInf,
          b + ": dimension 0");
}

/// A direction of the wrong dimension is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const EmptySet E(2);
    check(test::raises([&] { E.support_function(test::column({1.0, 2.0, 3.0})); },
                       Error::dimension_mismatch),
          "support_function: a direction of 3 dimensions");
    ChecksScope off(false);
    check(test::support(E, {1.0, 0.0}) == -kInf, "support_function: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_support_function_is_minus_infinity);
    checks_and_their_switch();
    return test::finish("empty_set support_function");
}
