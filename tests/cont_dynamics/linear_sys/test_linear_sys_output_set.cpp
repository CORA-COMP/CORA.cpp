// test_linear_sys_output_set - linear_sys output_set: y = C x is exact on zonotopes, and without C
// the output is the state

#include "cont_dynamics/linear_sys/linear_sys_testing.h"
#include "cont_dynamics/linear_sys/matlab_reference_inputs.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using namespace test::lin;

namespace {

struct Setup {
    Tensor A, B, C;
    Zonotope X0, U;
    Setup()
        : A(Tensor::from_data(matlab_reference::standard_inputs().A, {3, 3})),
          B(Tensor::from_data(matlab_reference::standard_inputs().B, {3, 2})),
          C(Tensor::from_data(matlab_reference::standard_inputs().C, {2, 3})),
          X0(Tensor::from_data(matlab_reference::standard_inputs().c0, {3, 1}),
             Tensor::from_data(matlab_reference::standard_inputs().G0, {3, 3})),
          U(Tensor::from_data(matlab_reference::standard_inputs().c_u, {2, 1}),
            Tensor::from_data(matlab_reference::standard_inputs().GU, {2, 3})) {}
};

/// The output set is C Z: its center is C c, its generators C G, and its support along d is the
/// support of the state set along C'd.
void maps_every_set_by_c(const std::string &b) {
    const Setup in;
    const LinearSys sys(in.A, in.B, in.C);
    const ReachSet R = sys.reach({.initial_set = in.X0, .input_set = in.U, .t_final = 1.0},
                                 alg::Standard{.time_step = 0.2, .taylor_terms = 6});
    const ReachSet Y = sys.output_set(R);
    check(Y.time_interval.size() == R.time_interval.size() &&
              Y.time_point.size() == R.time_point.size(),
          b + ": the number of sets");
    const std::vector<double> d{0.7, -1.1};
    const Tensor Ctd = in.C.transpose().matmul(test::column(d));
    for (std::size_t k = 0; k < R.time_interval.size(); ++k) {
        check(close(Y.time_interval[k].center(), in.C.matmul(R.time_interval[k].center())),
              b + ": the center of a step");
        check(close(Y.time_interval[k].support_function(test::column(d)),
                    R.time_interval[k].support_function(Ctd)),
              b + ": the support of a step along C'd");
    }
    for (std::size_t k = 0; k < R.time_point.size(); ++k)
        check(close(Y.time_point[k].generators(), in.C.matmul(R.time_point[k].generators())),
              b + ": a time point");
    check(Y.time_interval[0].dim() == 2, b + ": the output has p dimensions");
}

void is_the_state_without_c(const std::string &b) {
    const Setup in;
    const LinearSys sys(in.A, in.B);
    const ReachSet R = sys.reach({.initial_set = in.X0, .input_set = in.U, .t_final = 0.6},
                                 alg::Standard{.time_step = 0.2, .taylor_terms = 6});
    const ReachSet Y = sys.output_set(R);
    check(close(Y.time_interval[1].center(), R.time_interval[1].center()) &&
              close(Y.time_point[2].generators(), R.time_point[2].generators()),
          b + ": without C the output is the state");
}

/// Sets of another dimension than the states are refused; with the checks off, the same sets.
void checks_and_their_switch() {
    const LinearSys sys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}), Tensor({{1.0}, {0.0}}),
                        Tensor({{1.0, 1.0}}));
    const ReachSet R =
        sys.reach({.initial_set = Zonotope(Tensor({1.0, 1.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}})),
                   .t_final = 0.5},
                  alg::Standard{.time_step = 0.1});
    ReachSet wrong = R;
    wrong.time_point[0] =
        Zonotope(Tensor({1.0, 0.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    check(test::raises([&] { sys.output_set(wrong); }, Error::dimension_mismatch),
          "output_set: a set of three dimensions");
    const Tensor on = sys.output_set(R).time_point.back().center();
    ChecksScope off(false);
    check(test::close(sys.output_set(R).time_point.back().center(), on, 0),
          "output_set: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        maps_every_set_by_c(b);
        is_the_state_without_c(b);
    });
    checks_and_their_switch();
    return test::finish("linear_sys output_set");
}
