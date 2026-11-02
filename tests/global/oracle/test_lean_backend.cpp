// test_lean_backend - the lean Tensor backend, and with the oracles set its match with CORALean
//
// The backend rounds and orders its operations as CORALean does at dtype "nearest": a zonotope
// operation of CORA.cpp on it must give the oracle's numbers bit for bit.
//
// ORACLE: contSet/@zonotope/mtimes.m
// ORACLE: contSet/@zonotope/plus.m
// ORACLE: contSet/@zonotope/reduce.m
// ORACLE: contSet/@contSet/linComb.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"
#include "testing.h"

using namespace cora;
using test::throws;

int main() {
    const BackendScope lean_backend("lean");

    // sums nest to the right from zero, and the file is compiled without fused multiply-add
    const Tensor A({{1.0, 1e16, -1e16}});
    const Tensor x({{1.0}, {1.0}, {1.0}});
    test::check(A.matmul(x).data()[0] == 1.0, "1 + (1e16 + (-1e16 + 0)) keeps the one");
    test::check(throws([] { Tensor({{1.0}}).expm(); }), "expm is not available");
    test::check(throws([] { const BackendScope s("lean:cuda"); }), "no devices");

    if (test::oracle::available()) {
        const lean::DTypeScope nearest("nearest");
        test::oracle::Random rnd(7);
        for (const auto [n, m] : {std::pair{2, 3}, {5, 30}, {20, 200}}) {
            const std::string size = ", n=" + std::to_string(n) + ", m=" + std::to_string(m);
            const Tensor M = rnd.matrix(n, n);
            const Zonotope Z = rnd.zonotope(n, m), Z2 = rnd.zonotope(n, m);
            const lean::Zonotope L(Z), L2(Z2);

            // the same operation in CORA.cpp and in the oracle
            test::oracle::check_exact("mtimes" + size, Z.linear_map(M), L.linear_map(M));
            test::oracle::check_exact("plus" + size, Z.plus(Z2), L.plus(L2));
            test::oracle::check_exact("reduce" + size, Z.reduce(2), L.reduce(2));
            test::oracle::check_exact("linComb" + size, Z.linear_combination(Z2),
                                      L.linear_combination(L2));
        }
    }
    return test::finish("lean backend");
}
