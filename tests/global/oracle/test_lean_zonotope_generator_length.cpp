// test_lean_zonotope_generator_length - Zonotope::generator_length against the CORALean oracle
//
// The length of a generator is the root of the sum of its squares. At dtype nearest the oracle's
// root is the machine's, as CORA.cpp's on the lean backend: the lengths agree bit for bit.
//
// ORACLE: contSet/@zonotope/generatorLength.m
// Authors:       Claude Sonnet 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("zonotope.generatorLength", [](test::oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {5, 30}, {20, 200}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            // CORA.cpp's lengths are a row (1, m), the oracle's a column
            test::oracle::check(Z.generator_length().matrix_transpose(),
                                lean::Zonotope(Z).generator_length());
        }
    });
    return test::finish("lean zonotope generator length");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
