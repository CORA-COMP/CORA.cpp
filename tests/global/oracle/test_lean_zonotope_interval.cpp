// test_lean_zonotope_interval - Zonotope::interval against the CORALean oracle (its `box`)
//
// ORACLE: contSet/@zonotope/interval.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("zonotope.box", [](test::oracle::Random &rnd) {
        for (const auto [n, m] : {std::pair{2, 3}, {5, 30}, {20, 200}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            test::oracle::check(Z.interval(), lean::Zonotope(Z).interval());
        }
    });
    return test::finish("lean zonotope interval");
}
