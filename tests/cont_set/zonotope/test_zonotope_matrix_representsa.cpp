// test_zonotope_matrix_representsa - matrix zonotope representsa: the set as another type
//
// CORA: matrixSet/@matZonotope/representsa.m
// PROPERTY: matrixSet/@matZonotope/representsa.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/matrix_zonotope_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A point matrix is a point and, for the zero matrix, the origin; a set with a generator is
/// neither; a zero generator does not count.
void point_and_origin(const std::string &b) {
    const Zonotope point = test::point_set(test::matrix({1, 2, 3, 4}, 2, 2)),
                   zero = test::point_set(test::matrix({0, 0, 0, 0}, 2, 2));
    check(point.representsa("point"), b + ": a point matrix is a point");
    check(!point.representsa("origin"), b + ": a nonzero point is not the origin");
    check(zero.representsa("origin") && zero.representsa("point"), b + ": the zero matrix");
    check(!test::example().representsa("point"), b + ": the example is not a point");
    const Zonotope flat =
        Zonotope::matrix(test::matrix({0, 0, 0, 0}, 2, 2), Tensor::zeros({1, 2, 2}));
    check(flat.representsa("origin"), b + ": a zero generator does not count");
    check(!test::example().representsa("origin"), b + ": the example is not the origin");
    check(test::example().representsa("zonotope"), b + ": it is a zonotope");
}

/// A random set with nonzero generators is not a point; its center, or the set scaled by 0, is.
void random_sets(const std::string &b) {
    Rng rng(4);
    const Zonotope A = test::random_set(rng, 3, 2, 0.5);
    check(!A.representsa("point") && !A.representsa("origin"), b + ": a random set is no point");
    check(test::point_set(A.center()).representsa("point"), b + ": its center is a point");
    check((A * 0.0).representsa("point"), b + ": the set scaled by 0 is a point");
    check((A * 0.0).representsa("origin"), b + ": and the origin");
}

/// A point within tol of the origin is the origin; the default tolerance counts rounding as zero.
void tolerance(const std::string &b) {
    const Zonotope near = test::point_set(test::matrix({1e-9, 0, 0, 0}, 2, 2));
    check(!near.representsa("origin", 1e-12), b + ": 1e-9 is not zero for 1e-12");
    check(near.representsa("origin", 1e-6), b + ": it is for 1e-6");
}

/// A batch of sets and a type CORA does not know are refused; checks off: the same.
void checks_and_their_switch() {
    const Zonotope A = test::example();
    check(test::raises([&] { A.representsa("point", -1.0); }, Error::wrong_value),
          "representsa: tol");
    check(test::raises([&] { test::batch_of(A, A).representsa("point"); }, Error::not_supported),
          "representsa: a batch of sets");
    const bool on = A.representsa("point");
    ChecksScope off(false);
    check(A.representsa("point") == on, "representsa: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        point_and_origin(b);
        random_sets(b);
        tolerance(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope representsa");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
