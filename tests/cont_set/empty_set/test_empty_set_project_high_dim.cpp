// test_empty_set_project_high_dim - EmptySet project_high_dim: the empty set of the larger space
//
// CORA: contSet/@emptySet/projectHighDim_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_set_has_the_larger_dimension(const std::string &b) {
    check(EmptySet(2).project_high_dim(5, {3, 0}).dim() == 5, b + ": 2 into 5");
    check(EmptySet(2).project_high_dim(2, {1, 0}).dim() == 2, b + ": a permutation");
    check(EmptySet(0).project_high_dim(3, {}).dim() == 3, b + ": from R^0");
}

/// A smaller space, the wrong number of positions and a position outside are refused; with the
/// checks off the result is the same.
void checks_and_their_switch() {
    const EmptySet E(2);
    check(test::raises([&] { E.project_high_dim(1, {0, 0}); }, Error::wrong_value), "N < n");
    check(test::raises([&] { E.project_high_dim(4, {0}); }, Error::wrong_value), "one position");
    check(test::raises([&] { E.project_high_dim(4, {0, 4}); }, Error::wrong_value), "position 4");
    ChecksScope off(false);
    check(E.project_high_dim(4, {0, 3}).dim() == 4, "project_high_dim: the same with checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_set_has_the_larger_dimension);
    checks_and_their_switch();
    return test::finish("empty_set project_high_dim");
}
