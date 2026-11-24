// test_stl_model_check_trace - whether a sampled trace satisfies a formula
//
// CORA: specification/@stl/modelCheckTrace.m
// CORA authors: Niklas Kochdumper, Benedikt Seidl
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/settings.h"
#include "cora/specification/stl.h"
#include "stl_testing.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The trace (sin t, cos 3t) at the times 0, 0.1, ..., 2 (the cases were run in MATLAB's CORA).
void aux_trace(Tensor &x, std::vector<double> &t) {
    std::vector<double> d;
    for (int i = 0; i <= 20; ++i) {
        t.push_back(0.1 * i);
        d.push_back(std::sin(0.1 * i));
        d.push_back(std::cos(0.3 * i));
    }
    x = Tensor::from_data(d, {21, 2});
}

/// The formulas on traces, by the results of CORA.
void run(const std::string &b) {
    const auto v = StlExpr::variables("x", 2);
    const StlExpr &x1 = v[0], &x2 = v[1];
    Tensor x;
    std::vector<double> t;
    aux_trace(x, t);

    // the example of CORA's header: a quarter circle
    std::vector<double> d, tq;
    for (int i = 0; i <= 157; ++i) {
        const double phi = -M_PI / 2 + 0.01 * i;
        d.push_back(std::cos(phi)), d.push_back(std::sin(phi));
        tq.push_back(i / 157.0);
    }
    const Stl eq = (x2 < -0.5).until(x1 > 0.5, StlInterval(0, 1));
    check(eq.model_check_trace(Tensor::from_data(d, {158, 2}), tq), b + ": the header example");

    // the results MATLAB gives on the trace
    const Stl p = x1 > 0.8, q = x2 > -0.5;
    check((x1 < 0.5).model_check_trace(x, t), b + ": predicate");
    check(p.finally(StlInterval(0, 1)).model_check_trace(x, t), b + ": finally");
    check(!(x2 > -0.9).globally(StlInterval(0.5, 1.2)).model_check_trace(x, t), b + ": globally");
    check((x1 < 0.9).release(q, StlInterval(0, 1)).model_check_trace(x, t), b + ": release");
    check(!q.until(p, StlInterval(0.2, 1)).model_check_trace(x, t), b + ": until");
    check(!(x1 > 0.3).next(0.3).model_check_trace(x, t), b + ": next");
    check(((x1 + x2 > 0.0).globally(StlInterval(0, 0.3)))
              .finally(StlInterval(0, 1))
              .model_check_trace(x, t),
          b + ": nested");
    check(((x1 < 0.5) | (x2 < -0.8).finally(StlInterval(0, 0.5))).model_check_trace(x, t),
          b + ": or");
    check(((x1 < 0.5) | (x2 < 0.0)).globally(StlInterval(0, 1)).model_check_trace(x, t),
          b + ": globally of or");
    check((((x1 > 0.5) & (x2 > 0.0)) | (x1 < -0.2))
              .finally(StlInterval(0, 2))
              .model_check_trace(x, t),
          b + ": and, or");
    // a power, which CORA's own code cannot evaluate
    check((x1.power(2) < 0.7).globally(StlInterval(0, 1)).model_check_trace(x, t) == false,
          b + ": power");

    // invalid input and the checks off
    std::vector<double> bad = t;
    bad[3] += 0.01;
    check(test::raises([&] { p.model_check_trace(x, bad); }, Error::not_supported),
          b + ": not uniform");
    check(test::raises([&] { p.model_check_trace(x, {0.0, 1.0}); }, Error::wrong_value),
          b + ": sizes");
    {
        ChecksScope off(false);
        check(!Stl(false).model_check_trace(x, t), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_model_check_trace");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
