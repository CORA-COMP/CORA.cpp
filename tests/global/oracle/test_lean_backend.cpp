// test_lean_backend - the lean Tensor backend, and with CORACPP_ORACLE_UNSOUND set its match with CORALean
//
// The backend rounds and orders its operations as CORALean does at dtype "nearest": a zonotope
// operation of CORA.cpp on it must give the oracle's numbers bit for bit.

#include "contSet/zonotope/zonotope.h"
#include "global/oracle/oracle.h"
#include "global/oracle/zonotope.h"
#include "testing.h"

#include <cstdlib>
#include <cstring>
#include <random>

using namespace cora;
using test::throws;

namespace {

/// A tensor (rows, cols) of standard normal numbers, on the current backend.
Tensor aux_normal(std::mt19937_64 &gen, int64_t rows, int64_t cols) {
    std::normal_distribution<double> normal;
    std::vector<double> data(rows * cols);
    for (double &x : data) x = normal(gen);
    return Tensor::fromData(data, {rows, cols});
}

/// Whether two tensors hold the same bits (so -0 differs from 0).
bool aux_identical(const Tensor &a, const Tensor &b) {
    const std::vector<double> x = a.data(), y = b.data();
    return a.shape() == b.shape() &&
           std::memcmp(x.data(), y.data(), x.size() * sizeof(double)) == 0;
}

} // namespace

int main() {
    setBackend("lean");

    // sums nest to the right from zero, and the file is compiled without fused multiply-add
    const Tensor A({{1.0, 1e16, -1e16}});
    const Tensor x({{1.0}, {1.0}, {1.0}});
    test::check(A.matmul(x).data()[0] == 1.0, "1 + (1e16 + (-1e16 + 0)) keeps the one");
    test::check(throws([] { Tensor({{1.0}}).expm(); }), "expm is not available");
    test::check(throws([] { setBackend("lean:cuda"); }), "no devices");

    if (std::getenv("CORACPP_ORACLE_UNSOUND")) {
        lean::setDType("nearest");
        std::mt19937_64 gen(7);
        for (const auto [n, m] : {std::pair{2, 3}, {5, 30}, {20, 200}}) {
            const Tensor M = aux_normal(gen, n, n), c = aux_normal(gen, n, 1);
            const Tensor G = aux_normal(gen, n, m), c2 = aux_normal(gen, n, 1);
            const Tensor G2 = aux_normal(gen, n, m);
            const std::string size = "n=" + std::to_string(n) + ", m=" + std::to_string(m);
            const Zonotope Z(c, G), Z2(c2, G2);
            const lean::Zonotope L(c, G), L2(c2, G2);

            // the same operation in CORA.cpp and in the oracle
            const Zonotope mine = Z.mtimes(M);
            const Zonotope oracle = L.mtimes(lean::Tensor::from(M)).gather().first;
            test::check(aux_identical(mine.c, oracle.c) && aux_identical(mine.G, oracle.G),
                        "mtimes is identical, " + size);
            const Zonotope sum = Z.plus(Z2), sumOracle = L.plus(L2).gather().first;
            test::check(aux_identical(sum.c, sumOracle.c) && aux_identical(sum.G, sumOracle.G),
                        "plus is identical, " + size);
            const Zonotope red = Z.reduce(2), redOracle = L.reduce(2).gather().first;
            test::check(aux_identical(red.c, redOracle.c) && aux_identical(red.G, redOracle.G),
                        "reduce is identical, " + size);
            const Zonotope comb = Z.linComb(Z2), combOracle = L.linComb(L2).gather().first;
            test::check(aux_identical(comb.c, combOracle.c) && aux_identical(comb.G, combOracle.G),
                        "linComb is identical, " + size);
        }
        lean::setDType("binary64");
    }
    setBackend("eigen");
    return test::finish("lean backend");
}
