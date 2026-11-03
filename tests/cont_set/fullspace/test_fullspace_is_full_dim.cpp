// test_fullspace_is_full_dim - fullspace is_full_dim: the whole space has an interior
//
// CORA: contSet/@fullspace/isFullDim.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void is_full_dim_always() {
    check(Fullspace(1).is_full_dim() && Fullspace(4).is_full_dim() && Fullspace(0).is_full_dim(),
          "is_full_dim: in every dimension");
    check(Fullspace(2).is_full_dim(1e-3), "is_full_dim: with a tolerance");
    check(test::raises([] { Fullspace(2).is_full_dim(-1.0); }, Error::wrong_value),
          "is_full_dim: a negative tolerance");
}

/// With the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(Fullspace(2).is_full_dim(), "is_full_dim: the same with the checks off");
}

} // namespace

int main() {
    is_full_dim_always();
    checks_and_their_switch();
    return test::finish("fullspace is_full_dim");
}
