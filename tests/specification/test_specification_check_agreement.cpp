// test_specification_check_agreement - an unsafe polytope against the CORALean oracle, one way
//
// The oracle's `specification.check` proves an unsafe polytope of several rows clear only by a
// separating row; CORA.cpp decides exactly (a linear program in the polytope). So the answers are
// not bit-exact: whenever the oracle proves the set clear, CORA.cpp must say clear too.
//
// CORA: specification/@specification/check.m
// PROPERTY: specification/@specification/check.m
//
// Property: what the oracle proves disjoint from a polytope of 2 or 3 rows is disjoint (the
// operands come from oracle::run, Random(7))
//
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/oracle/specification.h"
#include "cora/specification/specification.h"
#include "global/oracle/oracle_testing.h"

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A specification A x <= b for the set S: the offsets are the support function of S along each row
/// moved by a random amount, so that the set is inside some of the halfspaces and outside others.
std::pair<Tensor, Tensor> aux_polytope(oracle::Random &rnd, const ContSet &S, int64_t n,
                                       int64_t p) {
    const Tensor A = rnd.matrix(p, n);
    std::vector<double> b(p);
    for (int64_t i = 0; i < p; ++i)
        b[i] = S.support_function(A.slice(-2, i, i + 1).matrix_transpose()).item() +
               rnd.uniform(-1.0, 1.0);
    return {A, Tensor::from_data(b, {p, 1})};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("specification.check", [](oracle::Random &rnd) {
        // the polytopes of 2 and 3 rows in 2 and 3 dimensions, a box and a zonotope each
        int proved = 0, total = 0;
        for (const int n : {2, 3}) {
            for (const int p : {2, 3}) {
                for (int trial = 0; trial < 8; ++trial) {
                    const Interval I = rnd.box(n);
                    const Zonotope Z = rnd.zonotope(n, n + 1);
                    for (const bool box : {true, false}) {
                        const ContSet &S = box ? static_cast<const ContSet &>(I) : Z;
                        // the offsets sit near the set, so the oracle proves some cases
                        const auto [A, b] = aux_polytope(rnd, S, n, p);
                        const Specification spec = Specification::unsafe_set(Polytope(A, b));
                        const lean::Tensor LA(A), Lb(b);
                        // the oracle's proof of 'clear'
                        const bool theirs =
                            box ? lean::check_specification("unsafeSet", LA, Lb, lean::Interval(I))
                                : lean::check_specification("unsafeSet", LA, Lb, lean::Zonotope(Z));
                        proved += theirs;
                        ++total;
                        test::check(!theirs || spec.check(S),
                                    "specification.check: a proved unsafe set is clear");
                    }
                }
            }
        }
        test::check(proved > 0 && proved < total,
                    "specification.check: the oracle proves some of the cases and not others");
    });
    return test::finish("specification check agreement");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
