// test_lean_linear_sys_closed_form_step - CORA's oneStep in closed form against CORALean
//
// One step of x' = A x + u, u in U + u0, from X0 in the closed form of CORA's oneStep
// (priv_closed_form_step) is CORALean's LinearSys.oneStepClosedFormFloat (the oracle's
// linearSys.oneStepClosedForm): the sets at the end of the step and over it agree bit for bit at
// nearest, for random systems of three states at the orders 2, 4 and 10 and of five at order 4,
// for the system of the oracle's self-test, for an input set without generators, and for systems
// whose exponential needs a raised order or no order below eta + 4096 encloses (the oracle
// answers null, the stepper is not admissible; beyond a scale of about 100 the powers of |A| dt
// overflow, which the oracle prints as a number out of range), and for stiff Robertson-like
// Jacobians (dt up to 0.01: from dt 0.02 on, the raised order lets the powers of A overflow, which
// the oracle prints out of range), where [e^{A dt}] is the order-16 enclosure at A dt / 2^s squared
// s times (the halvings s agree with CORALean's halvingsFloat, which is none only where no order is
// admissible either).
//
// ORACLE: contSet/@zonotope/mtimes.m
// CORA.cpp only: CORA evaluates oneStep with expm and A^-1; the closed form is CORALean's
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_closed_form_step.h"
#include "cora/global/oracle/linear_sys.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// One step of the system A from X under U and u, CORA.cpp's against the oracle's.
void aux_compare(const std::string &name, const Tensor &A, const Zonotope &X, const Zonotope &U,
                 const Tensor &u, double dt, int eta) {
    const std::optional<lean::Step> theirs =
        lean::LinearSys(A).one_step_closed_form(lean::Zonotope(X), lean::Zonotope(U), u, dt, eta);
    const ClosedFormStepper stepper(A, dt, eta);
    test::check(stepper.admissible() == theirs.has_value(), name + ": the same admissibility");
    if (!stepper.admissible() || !theirs) return;
    const ClosedFormStep mine = stepper.step(X, U, u);
    test::oracle::check(name + " end", mine.time_point, theirs->time_point);
    test::oracle::check(name + " window", mine.time_interval, theirs->time_interval);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("linearSys.oneStepClosedForm", [](test::oracle::Random &rnd) {
        // three states at the orders 2, 4, 10 and five at 4: the oracle's time grows with both
        for (const auto &[n, orders] :
             std::vector<std::pair<int64_t, std::vector<int>>>{{3, {2, 4, 10}}, {5, {4}}}) {
            const Tensor A = rnd.matrix(n, n);
            for (const int eta : orders)
                for (const double dt : {0.05, 0.2})
                    aux_compare("n" + std::to_string(n) + " eta" + std::to_string(eta) + " dt" +
                                    std::to_string(dt),
                                A, rnd.zonotope(n, 2), rnd.zonotope(n, 3), rnd.vector(n), dt, eta);
        }
    });
    // the system of the oracle's self-test, a point input, and an input set without generators
    test::oracle::run("linearSys.oneStepClosedForm", [](test::oracle::Random &rnd) {
        const Tensor A = Tensor({{-1.0, 0.0}, {0.5, -2.0}});
        for (const int eta : {2, 4, 10}) {
            aux_compare("self-test eta" + std::to_string(eta), A, rnd.zonotope(2, 3),
                        rnd.zonotope(2, 1), rnd.vector(2), 0.1, eta);
            aux_compare("no input generators eta" + std::to_string(eta), A, rnd.zonotope(2, 2),
                        Zonotope(Tensor({0.0, 0.0}), Tensor::zeros({2, 0})), rnd.vector(2), 0.1,
                        eta);
        }
    });
    // a stiff system: the order is raised past the one asked for, then none is admissible
    test::oracle::run("linearSys.oneStepClosedForm", [](test::oracle::Random &rnd) {
        for (const double scale : {5.0, 1e5}) {
            const Tensor A = rnd.matrix(3, 3) * scale;
            for (const int eta : {2, 4})
                aux_compare("stiff " + std::to_string(scale) + " eta" + std::to_string(eta), A,
                            rnd.zonotope(3, 2), rnd.zonotope(3, 2), rnd.vector(3), 1.0, eta);
        }
    });
    // the Jacobian of a Robertson reaction: stiff, so e^{|A| dt} is far wider than e^{A dt}
    test::oracle::run("linearSys.oneStepClosedForm", [](test::oracle::Random &rnd) {
        const Tensor A = Tensor({{-0.4, 0.0, 0.01}, {0.4, -3000.0, -0.01}, {0.0, 3000.0, 0.0}});
        for (const double dt : {1e-4, 1e-3, 1e-2})
            for (const int eta : {2, 4, 10})
                aux_compare("Robertson dt" + std::to_string(dt) + " eta" + std::to_string(eta), A,
                            rnd.zonotope(3, 2), rnd.zonotope(3, 2), rnd.vector(3), dt, eta);
    });
    // sparse generators (diagonal, with a zero column), as the passes of the nonlinear steps have
    test::oracle::run("linearSys.oneStepClosedForm", [](test::oracle::Random &rnd) {
        const Tensor A = Tensor({{-0.4, 0.0, 0.01}, {0.4, -30.0, -0.01}, {0.0, 30.0, 0.0}});
        const Tensor G = Tensor({{1e-3, 0.0, 0.0}, {0.0, 1e-4, 0.0}, {0.0, 0.0, 0.0}});
        const Tensor H = Tensor({{0.0, 0.0, 0.0}, {0.0, 2e-5, 0.0}, {0.0, 0.0, 3e-5}});
        for (const int eta : {2, 4})
            aux_compare("sparse eta" + std::to_string(eta), A, Zonotope(Tensor({1.0, 0.0, 0.0}), G),
                        Zonotope(rnd.vector(3), H), rnd.vector(3), 0.01, eta);
    });
    // the halvings s: norms up to 40, beyond which the oracle's powers overflow (file docstring)
    test::oracle::run("linearSys.oneStepClosedForm", [](test::oracle::Random &rnd) {
        for (const double scale : {0.01, 0.3, 1.0, 10.0, 40.0}) {
            const Tensor A = rnd.matrix(3, 3) * scale;
            aux_compare("scaled " + std::to_string(scale), A, rnd.zonotope(3, 2),
                        rnd.zonotope(3, 2), rnd.vector(3), 0.1, 4);
        }
    });
    return test::finish("lean linearSys closed-form step");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
