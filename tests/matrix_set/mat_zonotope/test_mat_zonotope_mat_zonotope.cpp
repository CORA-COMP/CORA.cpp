// test_mat_zonotope_mat_zonotope - MatZonotope: the constructors and their checks
//
// CORA: matrixSet/@matZonotope/matZonotope.m
// PROPERTY: matrixSet/@matZonotope/matZonotope.m

#include "mat_zonotope_testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;

namespace {

/// The forms of the constructor: center and generators, a center alone (no generators), and none
/// (the empty 0 x 0 set); the parts come back as given.
void constructor_forms(const std::string &b) {
    const MatZonotope A = test::example();
    check(A.rows() == 2 && A.cols() == 2 && A.num_generators() == 2, b + ": the example's sizes");
    check(close(A.generators(), {1, 0, 0, 1, 0, 1, 1, 0}), b + ": the generators, one matrix each");
    const MatZonotope point(test::matrix({1, 2, 3, 4, 5, 6}, 2, 3));
    check(point.rows() == 2 && point.cols() == 3 && point.num_generators() == 0,
          b + ": a point matrix has no generators");
    check(point.generators().shape() == std::vector<int64_t>({0, 2, 3}), b + ": its empty generators");
    const MatZonotope empty;
    check(empty.rows() == 0 && empty.cols() == 0 && empty.num_generators() == 0,
          b + ": the empty set is 0 x 0");
}

/// Every matrix C + sum b_i G_i with |b| <= 1 is in the set: the members' stacked points lie in the
/// zonotope of the stacking, and the corners of the cube are members.
void members_are_the_combinations(const std::string &b) {
    Rng rng(3);
    const MatZonotope A = test::random_set(rng, 3, 3, 0.5);
    bool inside = true;
    for (int trial = 0; trial < 20; ++trial)
        inside = inside && A.contains(test::member(A, test::coefficients(rng, 3)), "exact", 1e-9);
    check(inside, b + ": the combinations with |b| <= 1 are members");
    check(!A.contains(test::member(A, {1.5, 0.0, 0.0}), "exact", 1e-9), b + ": b = 1.5 is outside");
}

/// A batch of sets is one set with leading dimensions.
void batches(const std::string &b) {
    const MatZonotope A = test::example();
    const MatZonotope batch = test::batch_of(A, A.transpose());
    check(batch.rows() == 2 && batch.num_generators() == 2, b + ": the sizes of a batch");
    check(test::same_set(test::batch_member(batch, 1), A.transpose()), b + ": a member of the batch");
}

/// Centers and generators of another shape, a missing tensor, and NaN are refused; with the checks
/// off the same valid construction.
void checks_and_their_switch() {
    const Tensor C = test::matrix({1, 2, 3, 4}, 2, 2), G = Tensor::from_data({1, 0, 0, 1}, {1, 2, 2});
    check(test::raises([&] { MatZonotope(Tensor(), G); }, Error::wrong_input_in_constructor),
          "MatZonotope: an undefined center");
    check(test::raises([&] { MatZonotope(C, test::matrix({1, 0, 0, 1}, 2, 2)); },
                       Error::wrong_input_in_constructor),
          "MatZonotope: generators without the generator axis");
    check(test::raises([&] { MatZonotope(C, Tensor::from_data({1, 0, 0, 1, 0, 0}, {1, 2, 3})); },
                       Error::wrong_input_in_constructor),
          "MatZonotope: generators of another size than the center");
    check(test::raises([&] { MatZonotope(test::matrix({1, 2}, 1, 2).reshape({2})); },
                       Error::wrong_input_in_constructor),
          "MatZonotope: a center that is not a matrix");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(test::raises([&] { MatZonotope(test::matrix({1, nan, 3, 4}, 2, 2)); }, Error::wrong_value),
          "MatZonotope: a NaN center");
    check(test::raises([&] { MatZonotope(C, Tensor::from_data({1, nan, 0, 1}, {1, 2, 2})); },
                       Error::wrong_value),
          "MatZonotope: a NaN generator");
    ChecksScope off(false);
    check(close(MatZonotope(C, G).center(), C, 0), "MatZonotope: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        constructor_forms(b);
        batches(b);
    });
    test::for_each_backend(members_are_the_combinations);
    checks_and_their_switch();
    return test::finish("mat_zonotope constructor");
}
