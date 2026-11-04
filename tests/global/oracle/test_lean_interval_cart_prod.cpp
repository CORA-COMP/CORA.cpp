// test_lean_interval_cart_prod - Interval::cart_prod against the CORALean oracle (its `cartProd`)
//
// ORACLE: contSet/@interval/cartProd_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.cartProd", [](test::oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {5, 1}, {20, 7}}) {
            const Interval I = rnd.box(n), J = rnd.box(m);
            test::oracle::check(I.cart_prod(J), lean::Interval(I).cart_prod(lean::Interval(J)));
        }
    });
    return test::finish("lean interval cart_prod");
}
