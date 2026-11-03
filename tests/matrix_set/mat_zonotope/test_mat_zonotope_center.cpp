// test_mat_zonotope_center - MatZonotope center: the center matrix
//
// CORA: matrixSet/@matZonotope/center.m
// PROPERTY: matrixSet/@matZonotope/center.m

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// The center of the example is C; it is the member without any generator contribution, and a
/// batch keeps one center per set.
void center_is_the_member_at_zero(const std::string &b) {
    const MatZonotope A = test::example();
    check(close(A.center(), {1, 2, 3, 4}), b + ": the center of the example");
    Rng rng(1);
    const MatZonotope R = test::random_set(rng, 3, 2, 0.4);
    check(close(test::member(R, {0.0, 0.0}), R.center()), b + ": the member at b = 0 is the center");
    const MatZonotope batch = test::batch_of(A, A.plus(test::matrix({1, 1, 1, 1}, 2, 2)));
    check(close(batch.center().slice(0, 1, 2).squeeze(0), {2, 3, 4, 5}), b + ": a batch");
}

/// center has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const MatZonotope A = test::example();
    const Tensor on = A.center();
    ChecksScope off(false);
    check(close(A.center(), on, 0), "center: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean(center_is_the_member_at_zero);
    checks_and_their_switch();
    return test::finish("mat_zonotope center");
}
