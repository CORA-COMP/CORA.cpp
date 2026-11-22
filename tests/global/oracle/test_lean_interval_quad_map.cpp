// test_lean_interval_quad_map - Interval::quad_map against the CORALean oracle, bit for bit
//
// The oracle's `interval.quadMapCora` is the box of the quadratic map of the box's zonotope, the
// map being CORA's own (`Zonotope.quadMapTri`), as CORA.cpp computes it. `interval.quadMap` goes
// through a polynomial zonotope, a wider box, and is compared in
// test_lean_interval_quad_map_enclosure.
//
// ORACLE: contSet/@interval/quadMap.m
// Authors:       Claude Sonnet 5.5
// Written:       22-November-2026
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
    test::oracle::run("interval.quadMapCora", [](test::oracle::Random &rnd) {
        for (const int n : {1, 2, 4}) {
            const Interval I = rnd.box(n);
            const std::vector<Tensor> Q{rnd.matrix(n, n), rnd.matrix(n, n)};
            std::vector<lean::Tensor> LQ;
            LQ.reserve(Q.size());
            for (const Tensor &M : Q) LQ.push_back(lean::Tensor::from(M));
            test::oracle::check(I.quad_map(Q), lean::Interval(I).quad_map_cora(LQ));
        }
    });
    return test::finish("lean interval quad_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
