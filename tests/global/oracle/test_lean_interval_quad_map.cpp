// test_lean_interval_quad_map - Interval::quad_map against the CORALean oracle (its `quadMap`)
//
// CORA: contSet/@interval/quadMap.m
//
// No ORACLE tag: CORALean builds the box through a polynomial zonotope, a different (wider)
// enclosure than CORA's zonotope map, so a bit-exact test cannot exist. CORA.cpp's box must lie
// inside the oracle's sound one.

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.quadMap", [](test::oracle::Random &rnd) {
        for (const int n : {1, 2, 4}) {
            const Interval I = rnd.box(n);
            const std::vector<Tensor> Q{rnd.matrix(n, n), rnd.matrix(n, n)};
            std::vector<lean::Tensor> LQ;
            for (const Tensor &M : Q) LQ.push_back(lean::Tensor::from(M));
            test::oracle::check_enclosed("interval.quadMap", I.quad_map(Q), lean::Interval(I).quad_map(LQ));
        }
    });
    return test::finish("lean interval quad_map");
}
