// test_lean_interval_boundary_point - Interval::boundary_point against the CORALean oracle
//
// ORACLE: contSet/@interval/boundaryPoint.m
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.boundaryPoint", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            const Tensor dir = rnd.vector(n), start = I.center();
            // the oracle's answer is a box around the point; the point is in it
            const Tensor x = I.boundary_point(dir, start);
            const std::optional<lean::Interval> box = lean::Interval(I).boundary_point(
                lean::Tensor::from(dir), lean::Tensor::from(start));
            test::oracle::check(std::optional<Interval>(Interval(x, x)), box);
        }
    });
    return test::finish("lean interval boundary_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
