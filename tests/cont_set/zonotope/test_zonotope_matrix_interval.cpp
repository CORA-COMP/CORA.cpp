// test_zonotope_matrix_interval - matrix zonotope interval: the box around the set
//
// CORA: matrixSet/@matZonotope/intervalMatrix.m
// PROPERTY: matrixSet/@matZonotope/intervalMatrix.m
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

/// C +- (|G1| + |G2|) of the example: [[1, 1], [1, 1]] around [[1, 2], [3, 4]].
void box_of_the_example(const std::string &b) {
    const Interval box = test::example().interval();
    check(close(box.infimum(), {0, 1, 2, 3}), b + ": the lower bound");
    check(close(box.interval().sup(), {2, 3, 4, 5}), b + ": the upper bound");
    const Interval point = test::point_set(test::matrix({1, -2}, 1, 2)).interval();
    check(close(point.infimum(), {1, -2}) && close(point.interval().sup(), {1, -2}),
          b + ": a point matrix is its own box");
}

/// Every member is in the box, and the box is tight: each bound is reached at a corner of the cube.
void box_encloses_and_is_tight(const std::string &b) {
    Rng rng(7);
    const Zonotope A = test::random_set(rng, 3, 4, 0.5);
    const Interval box = A.interval();
    bool inside = true;
    for (int trial = 0; trial < 30; ++trial)
        inside = inside && box.contains(test::member(A, test::coefficients(rng, 4)), 1e-12);
    check(inside, b + ": every member is in the box");
    // the corner b_i = sign(G_i[0, 0]) maximizes the entry (0, 0)
    std::vector<double> corner;
    for (const double g : A.generators().slice(-1, 0, 1).slice(-2, 0, 1).data())
        corner.push_back(g >= 0 ? 1.0 : -1.0);
    const Tensor reached = test::member(A, corner), bound = box.interval().sup();
    check(close(reached.slice(0, 0, 1).slice(1, 0, 1), bound.slice(0, 0, 1).slice(1, 0, 1), 1e-12),
          b + ": the upper bound is attained");
    const Zonotope batch = test::batch_of(A, A.transpose());
    check(close(batch.interval().center().slice(0, 1, 2).squeeze(0), A.transpose().center(), 1e-12),
          b + ": a batch");
}

/// interval_matrix takes no input; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope A = test::example();
    const Tensor on = A.interval().infimum();
    ChecksScope off(false);
    check(close(A.interval().infimum(), on, 0), "interval_matrix: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        box_of_the_example(b);
        box_encloses_and_is_tight(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope interval_matrix");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
