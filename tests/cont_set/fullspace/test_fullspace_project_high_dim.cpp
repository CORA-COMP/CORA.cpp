// test_fullspace_project_high_dim - fullspace project_high_dim: only into the same dimension
//
// CORA: contSet/@fullspace/projectHighDim_.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void only_the_same_dimension_is_defined(const std::string &b) {
    check(Fullspace(2).project_high_dim(2, {1, 0}).dim() == 2, b + ": a permutation of R^2");
    check(Fullspace(0).project_high_dim(0, {}).dim() == 0, b + ": R^0");
}

/// A larger space cannot hold the new coordinates at 0 (Error::not_defined), and the arguments of
/// lift are checked; with the checks off the same-dimension result is the same.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([&] { F.project_high_dim(4, {0, 3}); }, Error::not_defined), "N = 4");
    check(test::raises([&] { F.project_high_dim(1, {0, 0}); }, Error::wrong_value), "N < n");
    check(test::raises([&] { F.project_high_dim(2, {0}); }, Error::wrong_value), "one position");
    check(test::raises([&] { F.project_high_dim(2, {0, 2}); }, Error::wrong_value), "position 2");
    ChecksScope off(false);
    check(F.project_high_dim(2, {0, 1}).dim() == 2, "project_high_dim: the same with checks off");
}

} // namespace

int main() {
    test::for_each_backend(only_the_same_dimension_is_defined);
    checks_and_their_switch();
    return test::finish("fullspace project_high_dim");
}
