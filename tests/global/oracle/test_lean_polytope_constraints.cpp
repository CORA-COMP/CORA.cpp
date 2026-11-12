// test_lean_polytope_constraints - the halfspaces a Polytope holds, built from a box and by the
//
// intersection of two, against the CORALean oracle
//
// ORACLE: contSet/@polytope/polytope.m
// ORACLE: contSet/@polytope/constraints.m
//
// The constructor of the halfspace form and its accessor constraints(), bit for bit, on the sets
// the benchmarks build: the polytope of a box (reach-avoid specifications, hybrid guards) and the
// rows that the intersection of two halfspace polytopes stacks. The facets of a vertex polytope
// (CORALean's VPolytope.hPolytope, qhull in CORA) have no oracle op and are not compared.
//
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/oracle/polytope.h"
#include "oracle_testing.h"

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The four parts of the constraints against the oracle's polytope, all of them reported.
void aux_check_constraints(const Polytope &mine, const lean::Polytope &theirs) {
    const Polytope other = theirs.nominal();
    const Polytope::Constraints C = mine.constraints();
    const Polytope::Constraints D = other.constraints();
    oracle::check("A", C.A, D.A);
    oracle::check("b", C.b, D.b);
    oracle::check("Ae", C.Ae, D.Ae);
    oracle::check("be", C.be, D.be);
}

/// The box |x| <= 2 and m random rows with offset 3: nonempty and bounded.
Polytope aux_halfspaces(oracle::Random &rnd, int64_t n, int64_t m) {
    const Tensor eye = Tensor::eye(n), negative = eye * 0.0 - eye;
    return {Tensor::concat({eye, negative, rnd.matrix(m, n)}, 0),
            Tensor::concat({Tensor::ones({2 * n, 1}) * 2.0, Tensor::ones({m, 1}) * 3.0}, 0)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;

    // the polytope of a box: two rows per dimension
    oracle::run("interval.polytope", [](oracle::Random &rnd) {
        for (const int n : {1, 2, 4}) {
            const Interval I = rnd.box(n);
            aux_check_constraints(Polytope(I), lean::Polytope::from_interval(I));
        }
    });

    // the rows of the intersection are the rows of both
    oracle::run("polytope.and", [](oracle::Random &rnd) {
        const Polytope a = aux_halfspaces(rnd, 3, 2), b = aux_halfspaces(rnd, 3, 4);
        aux_check_constraints(a.intersect(b), lean::Polytope(a).intersect(b));
    });
    return test::finish("lean polytope constraints");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
