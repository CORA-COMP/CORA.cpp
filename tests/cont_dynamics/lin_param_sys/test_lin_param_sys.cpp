// test_lin_param_sys - LinParamSys: the constructor, its checks and its parts
//
// The matrix set, B and c are kept; c defaults to zero; the dimensions are those of A and B.
// Shapes that do not fit are refused, and with the checks off the system is the same.
//
// CORA: contDynamics/@linParamSys/linParamSys.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/lin_param_sys/lin_param_sys.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The set {[-1 2; -2 -1] + b [0.1 0; 0 0.1] | b in [-1, 1]}.
MatZonotope aux_set() {
    return MatZonotope(Tensor({{-1.0, 2.0}, {-2.0, -1.0}}),
                       Tensor({{0.1, 0.0}, {0.0, 0.1}}).expand_dims(0));
}

/// The parts of a system with and without B and c.
void construction() {
    const LinParamSys plain(aux_set());
    check(plain.dim() == 2 && plain.input_dim() == 0 && !plain.B() && close(plain.c(), {0, 0}, 0),
          "constructor: without B and c");
    check(plain.type() == ParamType::constant, "constructor: constant parameters by default");
    const LinParamSys full(aux_set(), Tensor({{0.0}, {1.0}}), column({0.5, 0.0}),
                           ParamType::varying);
    check(full.input_dim() == 1 && close(full.c(), {0.5, 0}) && full.type() == ParamType::varying,
          "constructor: B, c and varying parameters");
    check(close(full.A().center(), {-1, 2, -2, -1}) && full.A().num_generators() == 1,
          "constructor: the matrix set");
}

/// Shapes that do not fit are refused; with the checks off the system is the same.
void checks_and_their_switch() {
    const MatZonotope wide(Tensor::zeros({2, 3}));
    check(test::raises([&] { LinParamSys{wide}; }, Error::wrong_input_in_constructor),
          "constructor: A must be square");
    check(test::raises([] { LinParamSys(aux_set(), Tensor::zeros({3, 1})); },
                       Error::wrong_input_in_constructor),
          "constructor: B with the rows of A");
    check(test::raises([] { LinParamSys(aux_set(), std::nullopt, column({1.0})); },
                       Error::wrong_input_in_constructor),
          "constructor: c with the rows of A");
    ChecksScope off(false);
    const LinParamSys same(aux_set(), Tensor({{0.0}, {1.0}}));
    check(same.dim() == 2 && same.input_dim() == 1, "constructor: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    construction();
    checks_and_their_switch();
    return test::finish("lin_param_sys");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
