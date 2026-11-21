// test_lean_capsule_support_function - Capsule::support_function against the CORALean oracle
//
// The support function of a capsule is d'c + |d'g| + r |d|. At dtype nearest the oracle takes the
// machine's square root, as CORA.cpp on the lean backend: the value agrees bit for bit.
//
// ORACLE: contSet/@capsule/supportFunc_.m
// Authors:       Claude Sonnet 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/global/oracle/capsule.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("capsule.supportFunc", [](test::oracle::Random &rnd) {
        for (const int n : {2, 3, 5}) {
            const Capsule C = rnd.capsule(n);
            const Tensor d = rnd.direction(n);
            test::oracle::check(C.support_function(d), lean::Capsule(C).support_function(d));
        }
    });
    return test::finish("lean capsule support function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
