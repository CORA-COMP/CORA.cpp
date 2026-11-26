// test_con_poly_zono_cart_prod - the product holds (x0, x0)
//
// CORA: contSet/@conPolyZono/cartProd_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_poly_zono_testing.h"

using namespace cora;
using namespace test::cpz;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &bk) {
        const ConPolyZono S = known();
        const std::vector<double> x0 = point(S, kA0, {kBeta0});
        check(in_box(S.cart_prod(S), {x0[0], x0[1], x0[0], x0[1]}) && S.cart_prod(S).dim() == 4,
              bk + ": cart_prod");
    });
    check(test::raises([] { (void)known().cart_prod(ConPolyZono::origin(0)); }, Error::wrong_value),
          "cart_prod: a set of no dimension");
    {
        ChecksScope off(false);
        (void)known().cart_prod(ConPolyZono::origin(1));
    }
    return test::finish("con_poly_zono cart_prod");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
