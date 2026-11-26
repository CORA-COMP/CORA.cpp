// test_con_poly_zono - ConPolyZono: the constructors, the conversions and the predicates
//
// The known set (con_poly_zono_testing.h) has its fields, its center and a factor point that meets
// its row; the conversions of a box, the origin and the empty set are what they say, isequal and
// representsa decide on the stored data, and a wrong operand is refused. The bit-exact rows are
// tests/global/oracle/test_lean_con_poly_zono*.cpp; one file per operation tests the others.
//
// CORA: contSet/@conPolyZono/conPolyZono.m
// CORA: contSet/@conPolyZono/origin.m
// CORA: contSet/@conPolyZono/empty.m
// CORA: contSet/@conPolyZono/dim.m
// CORA: contSet/@conPolyZono/center.m
// CORA: contSet/@conPolyZono/isequal.m
// CORA: contSet/@conPolyZono/representsa_.m
// CORA: contSet/@conPolyZono/display.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_poly_zono_testing.h"

using namespace cora;
using namespace test::cpz;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &bk) {
        const ConPolyZono S = known();
        check(S.dim() == 2 && close(S.center(), {1.0, 0.0}, 0), bk + ": the dimension, the center");
        check(S.constraints().contains(Tensor::from_data(kA0, {1, 2}), 1e-12),
              bk + ": a0 meets the row");
        check(S.ids() == std::vector<int64_t>{1, 2} && S.constraints().num_rows() == 1,
              bk + ": the identifiers and the rows");
        // the conversions
        check(in_box(ConPolyZono(Interval(test::vec({-1.0, 0.0}), test::vec({1.0, 2.0}))),
                     {1.0, 2.0}),
              bk + ": the box as a set");
        const Zonotope Z(test::vec({1.0, 2.0}), Tensor::from_data({1, 0, 0, 1}, {2, 2}));
        check(ConPolyZono(Z).generators().shape()[0] == 2 && ConPolyZono(Z).ids().size() == 2,
              bk + ": a zonotope's generators dependent");
        check(ConPolyZono(S.poly_zonotope()).constraints().num_rows() == 0,
              bk + ": a polynomial zonotope has no rows");
        // the predicates
        check(S.isequal(S) && !S.isequal(S + test::vec({1.0, 0.0})), bk + ": isequal");
        check(!S.representsa("point") && ConPolyZono::origin(2).representsa("point") &&
                  ConPolyZono::empty(2).representsa("emptySet"),
              bk + ": representsa");
        check(!S.display().empty(), bk + ": display");
    });
    check(
        test::raises([] { (void)known().plus(ConPolyZono::origin(3)); }, Error::dimension_mismatch),
        "plus: another dimension");
    check(test::raises([] { (void)known().representsa("zonotope"); }, Error::not_supported),
          "representsa: an unsupported type");
    return test::finish("con_poly_zono");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
