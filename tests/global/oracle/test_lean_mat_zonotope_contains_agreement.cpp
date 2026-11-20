// test_lean_mat_zonotope_contains_agreement - matrix zonotope contains against the CORALean
//
// oracle, as a decision on chosen inputs
//
// CORA.cpp solves a linear program per matrix, the oracle checks a certificate
// (Zonotope.containsMat: true is a proof, false refutes nothing): different algorithms, so this is
// no bit-exactness. They agree on members near the center (inside the box the certificate reaches)
// and on matrices outside the box of the set, and the oracle must certify the member.
//
// CORA: matrixSet/@matZonotope/contains.m
// CORA authors: Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       20-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/mat_zonotope.h"
#include "oracle_testing.h"

#include <array>
#include <vector>

using namespace cora;
using test::oracle::check;
using test::oracle::Random;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("matZonotope.contains", [&](Random &rnd) {
        for (const auto &[r, c, h] : std::vector<std::array<int64_t, 3>>{{2, 2, 8}, {2, 3, 12}}) {
            const Zonotope A = rnd.mat_zonotope(r, c, h);
            const Tensor G0 = A.generators().slice(-3, 0, 1).squeeze(-3);
            const Tensor G1 = A.generators().slice(-3, 1, 2).squeeze(-3);
            const Tensor member = A.center() + G0 * 0.0009765625 - G1 * 0.00048828125;
            const Tensor outside = A.interval().sup() + 1.0;
            for (const Tensor &M : {A.center(), member, outside})
                check(A.contains(M), lean::MatZonotope(A).contains({lean::Tensor::from(M)}));
            test::check(lean::MatZonotope(A).contains({lean::Tensor::from(member)}),
                        "matZonotope.contains: the member is certified");
        }
    });
    return test::finish("lean mat_zonotope contains agreement");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
