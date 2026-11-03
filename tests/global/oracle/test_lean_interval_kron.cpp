// test_lean_interval_kron - Interval::kron against the CORALean oracle (its `kron`)
//
// ORACLE: contSet/@interval/kron.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.kron", [](test::oracle::Random &rnd) {
        for (const auto [m, n, p, q] : {std::tuple{2, 3, 3, 2}, {1, 4, 2, 2}, {4, 4, 3, 1}}) {
            const Tensor A = rnd.matrix(m, n), B = rnd.matrix(p, q);
            const Interval I(A - 0.25, A + 0.5), J(B - 0.5, B + 0.25);
            test::oracle::check(I.kron(J), lean::Interval(I).kron(lean::Interval(J)));
        }
    });
    return test::finish("lean interval kron");
}
