// test_zonotope_matrix_reduce - matrix zonotope reduce: a matrix zonotope with fewer generators
//
// CORA: matrixSet/@matZonotope/reduce.m
// PROPERTY: matrixSet/@matZonotope/reduce.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/matrix_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: the reduction keeps the shape, has at most order * rows * cols generators and
/// contains the set (its box contains the box, and the members of the set are inside it).
void reduce_property(const std::string &b) {
    Rng rng(82);
    for (int order = 1; order <= 2; ++order) {
        const Zonotope A = test::random_set(rng, 2, 9, 1.0);
        const Zonotope R = A.reduce(order);
        check(R.center().size(-2) == 2 && R.center().size(-1) == 2 &&
                  R.nr_generators() <= order * 4,
              b + ": the shape and the number of generators");
        bool inside = true;
        for (int trial = 0; trial < 20; ++trial)
            inside =
                inside && R.contains(test::member(A, test::coefficients(rng, 9)), "exact", 1e-9);
        check(inside, b + ": the members are inside");
    }
    const Zonotope few = test::example();
    check(close(few.reduce(2).generators(), few.generators(), 0), b + ": few generators stay");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(reduce_property);
    check(test::raises([] { test::example().reduce(0); }, Error::wrong_value),
          "reduce: the order 0");
    ChecksScope off(false);
    check(test::example().reduce(1).center().size(-2) == 2, "reduce: the same with the checks off");
    return test::finish("mat_zonotope reduce");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
