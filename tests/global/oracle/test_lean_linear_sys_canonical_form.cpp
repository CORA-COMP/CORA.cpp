// test_lean_linear_sys_canonical_form - canonical_form and is_canonical_form against the oracle
//
// Bit for bit at nearest, inside the enclosure of the sound oracle.
//
// ORACLE: contDynamics/@linearSys/canonicalForm.m
// ORACLE: contDynamics/@linearSys/isCanonicalForm.m
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/oracle/linear_sys.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;

    // the input set is centered and mapped by B, the constant input is B (center + u_vec)
    test::oracle::run("linearSys.canonicalForm", [](test::oracle::Random &rnd) {
        const Tensor A = rnd.matrix(3, 3), B = rnd.matrix(3, 2), C = rnd.matrix(2, 3);
        const Zonotope U = rnd.zonotope(2, 4);
        const Tensor u_vec = rnd.matrix(2, 1);
        const CanonicalForm cf = LinearSys(A, B, C).canonical_form(U, u_vec);
        const lean::CanonicalForm reference = lean::LinearSys(A, B, C).canonical_form(U, u_vec);
        test::oracle::check(cf.U, reference.U);
        test::oracle::check(cf.u, reference.u);
    });

    // the identity matrices are canonical, a random B and a missing B are not
    test::oracle::run("linearSys.isCanonicalForm", [](test::oracle::Random &rnd) {
        const Tensor A = rnd.matrix(3, 3), I = A.eye_like(3), B = rnd.matrix(3, 3);
        test::oracle::check(LinearSys(A).is_canonical_form(),
                            lean::LinearSys(A).is_canonical_form());
        for (const Tensor &input : {I, B}) {
            test::oracle::check(LinearSys(A, input).is_canonical_form(),
                                lean::LinearSys(A, input, std::nullopt).is_canonical_form());
            test::oracle::check(LinearSys(A, input, I).is_canonical_form(),
                                lean::LinearSys(A, input, I).is_canonical_form());
            test::oracle::check(LinearSys(A, input, B).is_canonical_form(),
                                lean::LinearSys(A, input, B).is_canonical_form());
        }
    });
    return test::finish("lean linearSys canonical form");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
