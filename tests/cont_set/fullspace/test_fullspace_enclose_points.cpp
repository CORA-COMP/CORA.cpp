// test_fullspace_enclose_points - fullspace enclose_points: the whole space of the points'
//
// dimension
//
// CORA: contSet/@fullspace/enclosePoints.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void enclose_points_takes_the_number_of_rows(const std::string &b) {
    const Tensor points = Tensor::from_data({1, 2, 3, 4, 5, 6}, {3, 2});
    const Fullspace F = Fullspace::enclose_points(points);
    check(F.dim() == 3, b + ": three rows");
    check(F.contains(Interval(points, points)), b + ": the points are in it");
    check(Fullspace::enclose_points(Tensor::zeros({2, 3, 4})).dim() == 3,
          b + ": a batch of points");
}

/// Points that are no matrix are refused; the checks off give the same answer elsewhere.
void checks_and_their_switch() {
    check(test::raises([] { Fullspace::enclose_points(Tensor::zeros({3})); }, Error::wrong_value),
          "enclose_points: a flat array");
    ChecksScope off(false);
    check(Fullspace::enclose_points(Tensor::zeros({2, 2})).dim() == 2,
          "enclose_points: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(enclose_points_takes_the_number_of_rows);
    checks_and_their_switch();
    return test::finish("fullspace enclose_points");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
