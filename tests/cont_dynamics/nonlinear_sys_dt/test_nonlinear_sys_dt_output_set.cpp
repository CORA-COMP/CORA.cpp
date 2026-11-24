// test_nonlinear_sys_dt_output_set - the output set of nonlinearSysDT: a linear and a nonlinear g
//
// A linear output equation has no error and its set is the linear image, written out by hand; for
// the nonlinear one, sampled points of the states and the inputs map into the set of each tensor
// order.
//
// CORA: contDynamics/@contDynamics/outputSet.m
// PROPERTY: contDynamics/@contDynamics/outputSet.m
// CORA authors: Mark Wetzlinger, Laura Luetzow
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys_dt/nonlinear_sys_dt.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// x+ = x, with the output equation g.
NonlinearSysDT aux_system(const InputDynamics &g) {
    return NonlinearSysDT([](const std::vector<Expr> &x, const std::vector<Expr> &) { return x; },
                          2, 1, 0.1, g, 1);
}

/// y = 2 x0 - x1 + u on x in 1 + [-0.1, 0.1] x 0.5 + [-0.2, 0.2], u in [-1, 1]: [0.4, 2.6] ... by
/// hand.
void a_linear_output_has_no_error(const std::string &b) {
    const NonlinearSysDT sys =
        aux_system([](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{2 * x[0] - x[1] + u[0]};
        });
    const Zonotope X(test::vec({1.0, 0.5}), Tensor({{0.1, 0.0}, {0.0, 0.2}}));
    const Zonotope U(test::vec({0.0}), Tensor({{1.0}}));
    const DiscreteStep Y = sys.output_set(X, U, alg::DiscreteLinearization{});
    // 2 - 0.5 = 1.5 plus the radius 0.2 + 0.2 + 1
    const Interval I = Y.set.interval();
    check(sys.output_is_linear() && test::close(I.inf(), {1.5 - 1.4}, 1e-12) &&
              test::close(I.sup(), {1.5 + 1.4}, 1e-12),
          b + ": the linear image");
    check(test::close(Y.error.interval().sup(), {0.0}, 1e-15), b + ": no error");
}

/// y = x0 x1 + u x0: points of X and U map into the set, for both tensor orders.
void a_nonlinear_output_encloses(const std::string &b) {
    const NonlinearSysDT sys =
        aux_system([](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[0] * x[1] + u[0] * x[0]};
        });
    const Zonotope X(test::vec({1.0, 0.5}), Tensor({{0.1, 0.0}, {0.0, 0.2}}));
    const Zonotope U(test::vec({0.2}), Tensor({{0.1}}));
    for (int order : {2, 3}) {
        const DiscreteStep Y =
            sys.output_set(X, U, alg::DiscreteLinearization{.tensor_order_output = order});
        const Interval I = Y.set.interval();
        // 200 random points of the box of states and inputs, from a fixed seed
        bool inside = !sys.output_is_linear();
        Rng rng(4);
        std::vector<double> s(600);
        rng.uniform(s.data(), s.size(), -1.0, 1.0);
        for (int k = 0; k < 200; ++k) {
            const Tensor y = sys.output(test::vec({1.0 + 0.1 * s[3 * k], 0.5 + 0.2 * s[3 * k + 1]}),
                                        test::vec({0.2 + 0.1 * s[3 * k + 2]}));
            // the output of each point against the hull
            const double value = y.data()[0];
            inside =
                inside && value >= I.inf().data()[0] - 1e-12 && value <= I.sup().data()[0] + 1e-12;
        }
        // the verdict of this order
        check(inside, b + ": the points map into the set, order " + std::to_string(order));
    }
}

void wrong_arguments_and_checks_off(const std::string &b) {
    const NonlinearSysDT sys =
        aux_system([](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[0] * x[1] + u[0]};
        });
    const Zonotope X(test::vec({1.0, 0.5}), Tensor({{0.1, 0.0}, {0.0, 0.2}}));
    check(test::raises(
              [&] {
                  sys.output_set(X, std::nullopt,
                                 alg::DiscreteLinearization{.tensor_order_output = 4});
              },
              Error::wrong_value),
          b + ": tensor_order_output 4");
    const DiscreteStep on = sys.output_set(X, std::nullopt, alg::DiscreteLinearization{});
    ChecksScope off(false);
    // the same with the checks off
    const DiscreteStep no = sys.output_set(X, std::nullopt, alg::DiscreteLinearization{});
    check(test::close(on.set.center(), no.set.center(), 0), b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        a_linear_output_has_no_error(b);
        a_nonlinear_output_encloses(b);
        wrong_arguments_and_checks_off(b);
    });
    return test::finish("nonlinearSysDT output_set");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
