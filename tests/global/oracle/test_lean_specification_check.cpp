// test_lean_specification_check - Specification::check against the CORALean oracle (its
//
// `specification.check`)
//
// The specification is the polytope A x <= b (one halfspace per row), checked against a box and a
// zonotope. The oracle's true proves the set contained in it (safe set, invariant) or disjoint from
// it (unsafe set); false says that no proof was found, where CORA.cpp decides. A safe set and an
// unsafe halfspace are decided by their rows, so at nearest the answers are the same; an unsafe
// polytope the oracle proves clear only by a separating row, and CORA.cpp (exactly, by a linear
// program) must agree whenever it does.
//
// ORACLE: specification/@specification/check.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/oracle/specification.h"
#include "cora/specification/specification.h"
#include "oracle_testing.h"

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
        int proved = 0, total = 0;
        for (const int n : {2, 3}) {
            for (const int p : {1, 2, 3}) {
                for (int trial = 0; trial < 4; ++trial) {
                    const Interval I = rnd.box(n);
                    const Zonotope Z = rnd.zonotope(n, n + 1);
                    for (const char *kind : {"safeSet", "unsafeSet"}) {
                        for (const bool box : {true, false}) {
                            const ContSet &S = box ? static_cast<const ContSet &>(I) : Z;
                            const auto [A, b] = aux_polytope(rnd, S, n, p);
                            const bool safe = std::string(kind) == "safeSet";
                            const Polytope P(A, b);
                            const Specification spec =
                                safe ? Specification::safe_set(P) : Specification::unsafe_set(P);
                            const lean::Tensor LA(A), Lb(b);
                            // the oracle proves the verdict
                            const bool theirs =
                                box ? lean::check_specification(kind, LA, Lb, lean::Interval(I))
                                    : lean::check_specification(kind, LA, Lb, lean::Zonotope(Z));
                            proved += theirs;
                            ++total;
                            // only an unsafe polytope is proved by one row, CORA.cpp is exact
                            if (safe || p == 1)
                                oracle::check(spec.check(S), lean::Certified{theirs, theirs});
                            else
                                test::check(!theirs || spec.check(S),
                                            "specification.check: a proved unsafe set is clear");
                        }
                    }
                }
            }
        }
        test::check(proved > 0 && proved < total,
                    "specification.check: the oracle proves some of the cases and not others");
    });
    return test::finish("lean specification check");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
