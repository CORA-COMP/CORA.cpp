// test_zonotope_matrix - Zonotope::matrix: the constructor of a matrix zonotope and its checks
//
// CORA: matrixSet/@matZonotope/matZonotope.m
// PROPERTY: matrixSet/@matZonotope/matZonotope.m
// CORA authors: Matthias Althoff, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/matrix_zonotope_testing.h"


using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The constructor and a set of one matrix (no generators): the parts come back as given.
void constructor_forms(const std::string &b) {
    const Zonotope A = test::example();
    check(A.center().size(-2) == 2 && A.center().size(-1) == 2 && A.nr_generators() == 2,
          b + ": the example's sizes");
    check(close(A.generators(), {1, 0, 0, 1, 0, 1, 1, 0}), b + ": the generators, one matrix each");
    const Zonotope point = test::point_set(test::matrix({1, 2, 3, 4, 5, 6}, 2, 3));
    check(point.center().size(-2) == 2 && point.center().size(-1) == 3 &&
              point.nr_generators() == 0,
          b + ": a point matrix has no generators");
    check(point.generators().shape() == std::vector<int64_t>({0, 2, 3}),
          b + ": its empty generators");
}

/// Every matrix C + sum b_i G_i with |b| <= 1 is in the set: the members' stacked points lie in the
/// zonotope of the stacking, and the corners of the cube are members.
void members_are_the_combinations(const std::string &b) {
    Rng rng(3);
    const Zonotope A = test::random_set(rng, 3, 3, 0.5);
    bool inside = true;
    for (int trial = 0; trial < 20; ++trial)
        inside = inside && A.contains(test::member(A, test::coefficients(rng, 3)), "exact", 1e-9);
    check(inside, b + ": the combinations with |b| <= 1 are members");
    check(!A.contains(test::member(A, {1.5, 0.0, 0.0}), "exact", 1e-9), b + ": b = 1.5 is outside");
}

/// A batch of sets is one set with leading dimensions.
void batches(const std::string &b) {
    const Zonotope A = test::example();
    const Zonotope batch = test::batch_of(A, A.transpose());
    check(batch.center().size(-2) == 2 && batch.nr_generators() == 2, b + ": the sizes of a batch");
    check(test::same_set(test::batch_member(batch, 1), A.transpose()),
          b + ": a member of the batch");
}

/// Centers and generators of another shape and a missing tensor are refused; with the checks off
/// the same valid construction.
void checks_and_their_switch() {
    const Tensor C = test::matrix({1, 2, 3, 4}, 2, 2),
                 G = Tensor::from_data({1, 0, 0, 1}, {1, 2, 2});
    check(test::raises([&] { Zonotope::matrix(Tensor(), G); }, Error::wrong_input_in_constructor),
          "Zonotope: an undefined center");
    check(test::raises([&] { Zonotope::matrix(C, test::matrix({1, 0, 0, 1}, 2, 2)); },
                       Error::wrong_input_in_constructor),
          "Zonotope: generators without the generator axis");
    check(
        test::raises([&] { Zonotope::matrix(C, Tensor::from_data({1, 0, 0, 1, 0, 0}, {1, 2, 3})); },
                     Error::wrong_input_in_constructor),
        "Zonotope: generators of another size than the center");
    check(test::raises([&] { Zonotope::matrix(Tensor::zeros({2}), Tensor::zeros({0, 2})); },
                       Error::wrong_input_in_constructor),
          "Zonotope: a center that is not a matrix");
    ChecksScope off(false);
    check(close(Zonotope::matrix(C, G).center(), C, 0), "Zonotope: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        constructor_forms(b);
        batches(b);
    });
    test::for_each_backend(members_are_the_combinations);
    checks_and_their_switch();
    return test::finish("mat_zonotope constructor");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
