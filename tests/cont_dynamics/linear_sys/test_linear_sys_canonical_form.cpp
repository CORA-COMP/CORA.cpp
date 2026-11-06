// test_linear_sys_canonical_form - canonical_form: the input matrix moves into the sets
//
// The input set is centered and mapped by B; its center becomes the constant input.
//
// CORA: contDynamics/@linearSys/canonicalForm.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_steps_testing.h"

using namespace cora;
using namespace test::steps;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// B (U - c) and B (c + u_vec): the generators B G, the constant input of every column of u_vec.
void the_input_matrix_moves_into_the_sets(const std::string &b) {
    const Tensor B({{1.0, 2.0}, {0.0, 3.0}}), C({{1.0, 1.0}});
    const LinearSys sys(diagonal(), B, C);
    const Zonotope U(test::column({1.0, -1.0}), Tensor({{0.5, 0.0}, {0.0, 0.25}}));
    const CanonicalForm cf = sys.canonical_form(U, Tensor({{0.5, 0.0}, {0.5, 1.0}}));
    check(test::close(cf.U.center(), {0.0, 0.0}, 0), b + ": the set is centered");
    check(test::close(cf.U.generators(), {0.5, 0.5, 0.0, 0.75}, 1e-15), b + ": generators B G");
    // B (c + u_vec): the columns (1.5, -0.5) and (1, 0)
    check(test::close(cf.u, {0.5, 1.0, -1.5, 0.0}, 1e-15), b + ": a constant input per step");
    check(cf.sys.B().has_value() && test::close(*cf.sys.B(), Tensor::eye(2), 0),
          b + ": B is I now");
    check(cf.sys.C().has_value() && test::close(*cf.sys.C(), C, 0), b + ": C stays");
    // the default is no extra input
    check(test::close(sys.canonical_form(U).u, {-1.0, -3.0}, 1e-15), b + ": u_vec defaults to 0");
}

/// Without B, a set of the wrong dimension and a bad u_vec are refused; with the checks off the
/// same call works.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(diagonal(), Tensor::eye(2)), autonomous(diagonal());
    const Zonotope U = origin(2);
    check(test::raises([&] { autonomous.canonical_form(U); }, Error::wrong_value), b + ": no B");
    check(test::raises([&] { sys.canonical_form(origin(3)); }, Error::dimension_mismatch),
          b + ": U of three dimensions");
    check(test::raises([&] { sys.canonical_form(U, test::column({1.0, 2.0, 3.0})); },
                       Error::dimension_mismatch),
          b + ": u_vec of three rows");
    const Tensor on = sys.canonical_form(U, test::column({1.0, 2.0})).u;
    ChecksScope off(false);
    check(test::close(sys.canonical_form(U, test::column({1.0, 2.0})).u, on, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_input_matrix_moves_into_the_sets(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys canonical_form");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
