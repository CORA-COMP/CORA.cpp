// test_lean_mat_zonotope_norm - Zonotope::norm of a matrix zonotope against the CORALean oracle
//
// The norm is that of the enclosing interval matrix. The norms one and inf are sums and maxima of
// the same entries. The Frobenius norm is a square root, which at dtype nearest the oracle takes
// as the machine does, as CORA.cpp on the lean backend: all three agree bit for bit.
//
// ORACLE: matrixSet/@matZonotope/norm.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   21-November-2026
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "cora/global/oracle/mat_zonotope.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The norm of a set through ContSet::norm, which takes the type of a matrix norm.
Tensor set_norm(const ContSet &S, Tensor::Norm type) { return S.norm(type); }

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("matZonotope.norm", [](test::oracle::Random &rnd) {
        for (const auto &[r, c, h] : {std::array<int64_t, 3>{2, 2, 1}, {3, 4, 2}, {5, 1, 3}}) {
            const Zonotope A = rnd.mat_zonotope(r, c, h);
            const lean::MatZonotope L(A);
            test::oracle::check("matZonotope.norm 1", set_norm(A, Tensor::Norm::one), L.norm("1"));
            test::oracle::check("matZonotope.norm inf", set_norm(A, Tensor::Norm::inf),
                                L.norm("inf"));
            test::oracle::check("matZonotope.norm fro", set_norm(A, Tensor::Norm::fro),
                                L.norm("fro"));
        }
    });
    return test::finish("lean mat_zonotope norm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
