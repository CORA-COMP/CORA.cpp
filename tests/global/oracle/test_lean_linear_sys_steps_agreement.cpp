// test_lean_linear_sys_steps_agreement - the solutions of a step against the CORALean oracle
//
// The boxes agree to 1e-9 (relative), not bit for bit: CORALean bounds the remainder of the Taylor
// series by a norm and applies [e^{A dt}] as an interval matrix, where CORA takes elementwise
// remainders and the matrix e^{A dt}. With dt = 0.01 and 10 terms the remainders are 1e-20.
//
// CORA: contDynamics/@linearSys/homogeneousSolution.m
// CORA: contDynamics/@linearSys/particularSolution_timeVarying.m
// CORA: contDynamics/@linearSys/particularSolution_constant.m
// CORA: contDynamics/@linearSys/affineSolution.m
// CORA: contDynamics/@linearSys/oneStep.m
// CORA: contDynamics/@linearSys/post.m
// CORA: contDynamics/@linearSys/errorSolution_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/oracle/linear_sys.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kStep = 0.01;
constexpr int kTerms = 10;
constexpr double kAgreement = 1e-9;
constexpr double kGap = 1e-5;

/// A random system matrix and a set in its state space.
struct Operands {
    Tensor A;
    Zonotope X, U;
    Tensor u;
};

Operands draw(test::oracle::Random &rnd) {
    return {rnd.matrix(3, 3), rnd.zonotope(3, 4), rnd.zonotope(3, 2), rnd.vector(3)};
}

/// The step's enclosure lies in the oracle's, which bounds the remainders by norms; it is
/// wider by at most `gap` (relative).
void aux_inside(const Zonotope &mine, const lean::Zonotope &theirs, double gap) {
    const Interval box = mine.interval();
    const auto [inf, sup] = theirs.interval().gather();
    const std::vector<double> a = box.inf().data(), b = inf.data(), c = box.sup().data(),
                              d = sup.data();
    bool ok = a.size() == b.size();
    for (std::size_t i = 0; ok && i < a.size(); ++i) {
        const double slack = kAgreement * (1 + std::abs(b[i])), far = gap * (1 + std::abs(b[i]));
        ok = a[i] >= b[i] - slack && a[i] <= b[i] + far && c[i] <= d[i] + slack &&
             c[i] >= d[i] - far;
    }
    test::check(ok, test::oracle::op_name() + ": inside the oracle's box and close to it");
}

/// The body on the default backend against the oracle at nearest (the lean backend has no expm,
/// and these rows agree up to rounding, not bit for bit).
template <class F>
void aux_run(const std::string &op, F body) {
    if (!test::oracle::serves(op)) return;
    test::oracle::op_name() = op;
    const lean::DTypeScope dtype("nearest");
    test::oracle::Random rnd(7);
    body(rnd);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    using test::oracle::check_hull;

    // [e^{A dt}] X and the enclosure of the step, which the oracle calls post
    aux_run("linearSys.homogeneousSolution", [](test::oracle::Random &rnd) {
        const Operands o = draw(rnd);
        const HomogeneousSolution mine = LinearSys(o.A).homogeneous_solution(o.X, kStep, kTerms);
        const lean::LinearSys theirs(o.A);
        check_hull(mine.time_point, theirs.homogeneous_solution(o.X, kStep, kTerms), kAgreement);
        if (!test::oracle::serves("linearSys.post")) return;
        aux_inside(mine.time_interval, theirs.post(o.X, kStep, kTerms).time_interval, kGap);
    });

    // the terms of the series stay apart
    aux_run("linearSys.particularSolution_timeVarying", [](test::oracle::Random &rnd) {
        const Operands o = draw(rnd);
        check_hull(LinearSys(o.A).particular_solution_time_varying(o.U, kStep, kTerms),
                   lean::LinearSys(o.A).particular_solution_time_varying(o.U, kStep, kTerms),
                   kAgreement);
    });

    // for a point the closed form and the series agree; a set differs, CORALean keeps terms apart
    aux_run("linearSys.particularSolution_constant", [](test::oracle::Random &rnd) {
        const Operands o = draw(rnd);
        const Zonotope point(o.u, o.u.zeros_like({3, 0}));
        const ConstantSolution mine =
            LinearSys(o.A).particular_solution_constant(point, kStep, kTerms);
        check_hull(mine.time_point,
                   lean::LinearSys(o.A).particular_solution_constant(point, kStep, kTerms),
                   kAgreement);
        // CORALean's affine solution is the same sum of the input
        if (!test::oracle::serves("linearSys.affineSolution")) return;
        check_hull(LinearSys(o.A).affine_solution(o.X, o.u, kStep, kTerms).constant,
                   lean::LinearSys(o.A).affine_solution(point, kStep, kTerms), kAgreement);
    });

    // one step of reach: the end of the step and its enclosure
    aux_run("linearSys.oneStep", [](test::oracle::Random &rnd) {
        const Operands o = draw(rnd);
        const OneStep mine = LinearSys(o.A).one_step(o.X, o.U, o.u, kStep, kTerms);
        const lean::Step theirs = lean::LinearSys(o.A).one_step(o.X, o.U, o.u, kStep, kTerms);
        check_hull(mine.time_point, theirs.time_point, kAgreement);
        aux_inside(mine.time_interval, theirs.time_interval, kGap);
    });

    // the series of the input solution cut at a given order (CORA's tt_err)
    aux_run("linearSys.errorSolution_adaptive", [](test::oracle::Random &rnd) {
        const Operands o = draw(rnd);
        const Zonotope point(o.u, o.u.zeros_like({3, 0}));
        const LinearSys sys(o.A);
        const lean::LinearSys theirs(o.A);
        check_hull(sys.error_solution_adaptive(o.U, std::nullopt, kStep, 0.1, kTerms),
                   theirs.error_solution_adaptive(o.U, std::nullopt, kStep, kTerms), kAgreement);
        check_hull(sys.error_solution_adaptive(o.U, point, kStep, 0.1, kTerms),
                   theirs.error_solution_adaptive(o.U, point, kStep, kTerms), kAgreement);
    });
    return test::finish("lean linearSys steps");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
