// test_cont_set_center - ContSet::center and ContSet::dim: the center and the dimension of any set
//
// CORA: contSet/@contSet/center.m
// CORA: contSet/@contSet/dim.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// The center and the dimension through the base class, for a box, a zonotope and a batch.
void center_and_dim_of_any_set(const std::string &b) {
    const Interval I(column({-1.0, 0.0, 2.0}), column({1.0, 2.0, 4.0}));
    const Zonotope Z(column({1.0, 0.0}), Tensor::from_data({0.5, 0.0, 0.0, 0.5}, {2, 2}));
    const ContSet &A = I, &B = Z;
    check(test::close(A.center(), {0.0, 1.0, 3.0}), b + ": the center of a box");
    check(A.dim() == 3 && B.dim() == 2, b + ": the dimensions");
    check(test::close(B.center(), {1.0, 0.0}), b + ": the center of a zonotope");
    const Interval batch = Interval::stack({I, I});
    const ContSet &C = batch;
    check(C.center().shape() == std::vector<int64_t>({2, 3, 1}) && C.dim() == 3,
          b + ": a batch has a center per member");
    check(test::UnregisteredSet().dim() == 2, b + ": the dimension of a custom set");
}

/// Neither has an input to refuse; the checks off change nothing.
void checks_and_their_switch() {
    const Interval I(column({-1.0}), column({1.0}));
    ChecksScope off(false);
    check(test::close(static_cast<const ContSet &>(I).center(), {0.0}), "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(center_and_dim_of_any_set);
    checks_and_their_switch();
    return test::finish("cont_set center");
}
