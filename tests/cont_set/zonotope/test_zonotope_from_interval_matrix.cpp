// test_zonotope_from_interval_matrix - an interval matrix as a matrix zonotope
//
// CORA: matrixSet/@intervalMatrix/matZonotope.m
// CORA authors: Matthias Althoff, Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/interval_matrix_testing.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const Interval A = test::interval_matrix(test::matrix({1, 2, 3, 4}, 2, 2),
                                                 test::matrix({0.5, 0, 0, 1}, 2, 2));
        const Zonotope M = Zonotope::from_interval_matrix(A);
        check(M.nr_generators() == 4 && close(M.center(), {1, 2, 3, 4}, 0),
              b + ": a generator per entry");
        // generator j n + i holds the radius of entry (i, j) there (column-major)
        check(close(M.generators(), {0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, 0),
              b + ": the radius on the entry of each generator");
        check(close(M.interval().inf(), A.infimum(), 1e-15) &&
                  close(M.interval().sup(), A.supremum(), 1e-15),
              b + ": the box is the interval matrix");
    });
    check(test::raises([&] { Zonotope::from_interval_matrix(Interval::empty(2)); },
                       Error::not_supported),
          "from_interval: the empty interval has no matrix zonotope");
    ChecksScope off(false);
    check(Zonotope::from_interval_matrix(test::example()).nr_generators() == 4,
          "mat_zonotope: the same with the checks off");
    return test::finish("interval_matrix mat_zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
