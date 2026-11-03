// test_mat_zonotope_transpose - MatZonotope transpose: the transposed matrices
//
// CORA: matrixSet/@matZonotope/transpose.m
// PROPERTY: matrixSet/@matZonotope/transpose.m

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// The center and every generator are transposed: [[1, 2], [3, 4]] becomes [[1, 3], [2, 4]].
void transpose_of_the_example(const std::string &b) {
    const MatZonotope T = test::example().transpose();
    check(close(T.center(), {1, 3, 2, 4}), b + ": the center");
    check(close(T.generators(), {1, 0, 0, 1, 0, 1, 1, 0}), b + ": the generators (both symmetric)");
    const MatZonotope W = test::set_of({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, 2, 3).transpose();
    check(W.rows() == 3 && W.cols() == 2 && close(W.center(), {1, 4, 2, 5, 3, 6}),
          b + ": a 2 x 3 set");
}

/// The transpose of a member is a member of the transpose (exactly), and transposing twice is the
/// identity; a batch transposes per set.
void members_transpose(const std::string &b) {
    Rng rng(13);
    const MatZonotope A = test::random_set(rng, 3, 3, 0.5);
    const std::vector<double> coeff = test::coefficients(rng, 3);
    check(
        close(test::member(A.transpose(), coeff), test::member(A, coeff).matrix_transpose(), 1e-14),
        b + ": the transpose of a member");
    check(test::same_set(A.transpose().transpose(), A, 0), b + ": twice is the identity");
    check(test::same_set(test::batch_member(test::batch_of(A, A).transpose(), 0), A.transpose(), 0),
          b + ": a batch");
}

/// transpose takes no input; with the checks off the result is the same.
void checks_and_their_switch() {
    const MatZonotope A = test::example();
    const Tensor on = A.transpose().center();
    ChecksScope off(false);
    check(close(A.transpose().center(), on, 0), "transpose: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        transpose_of_the_example(b);
        members_transpose(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope transpose");
}
