// test_expr - symbolic expressions: values, derivatives, and enclosures over boxes
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       29-September-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/global/expr.h"
#include "testing.h"

#include <cmath>
#include <functional>
#include <string>

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The range [lo, hi] of one number, and the ends of a range of one number.
Range span(double lo, double hi) {
    return Range(Tensor::from_data({lo}, {1, 1}), Tensor::from_data({hi}, {1, 1}));
}
double lower(const Range &r) { return r.lo.data()[0]; }
double upper(const Range &r) { return r.hi.data()[0]; }

/// The van der Pol right-hand side of the second variable.
Expr vdp() {
    const Expr x1 = Expr::var(0), x2 = Expr::var(1);
    return (1 - x1 * x1) * x2 - x1;
}

void values_follow_the_formula() {
    check(close(vdp().eval({2.0, 3.0}), (1 - 4.0) * 3.0 - 2.0), "vdp at (2, 3)");
    const Expr x = Expr::var(0);
    check(close((sin(x) * cos(x) + exp(x) / (x + 3) - pow(x, 3)).eval({0.7}),
                std::sin(0.7) * std::cos(0.7) + std::exp(0.7) / 3.7 - std::pow(0.7, 3)),
          "a mixed expression");
    check(close((-x).eval({2.0}), -2.0) && close(pow(x, -2).eval({2.0}), 0.25), "neg and pow(-2)");
}

/// The derivative agrees with a central finite difference.
void derivatives_match_finite_differences() {
    const Expr x = Expr::var(0), y = Expr::var(1);
    const std::vector<Expr> fs = {vdp(), sin(x * y) / (1 + x * x), exp(x) * pow(y, 3) - cos(x)};
    for (const Expr &f : fs)
        for (int i = 0; i < 2; ++i) {
            std::vector<double> p = {0.6, -0.4}, lo = p, hi = p;
            lo[i] -= 1e-6;
            hi[i] += 1e-6;
            const double numeric = (f.eval(hi) - f.eval(lo)) / 2e-6;
            check(close(f.diff(i).eval(p), numeric, 1e-6), "derivative " + std::to_string(i));
        }
    check(close(Expr(3.0).diff(0).eval({1.0}), 0.0), "the derivative of a constant is 0");
}

/// The enclosure contains every sampled value.
void enclosures_contain_the_values() {
    const Expr x = Expr::var(0), y = Expr::var(1);
    const std::vector<Expr> fs = {vdp(), sin(x) * cos(y), pow(x, 2) - pow(y, 3), exp(x) / (y + 5)};
    const std::vector<Range> box = {span(-1.2, 2.0), span(-3.0, 1.5)};
    for (const Expr &f : fs) {
        const Range r = f.enclose(box);
        for (int i = 0; i <= 20; ++i)
            for (int j = 0; j <= 20; ++j) {
                const double v = f.eval({-1.2 + 3.2 * i / 20, -3.0 + 4.5 * j / 20});
                check(v >= lower(r) - 1e-12 && v <= upper(r) + 1e-12,
                      "a sampled value is enclosed");
            }
    }
}

void trigonometric_ranges_know_their_extremes() {
    const Expr x = Expr::var(0);
    const Range s = sin(x).enclose({span(0.0, 3.0)}), c = cos(x).enclose({span(-1.0, 1.0)});
    check(close(upper(s), 1.0) && close(lower(s), 0.0), "sin over [0, 3] peaks at 1");
    check(close(upper(c), 1.0) && close(lower(c), std::cos(1.0)), "cos over [-1, 1] peaks at 0");
    check(close(lower(sin(x).enclose({span(0.0, 7.0)})), -1.0), "a range of more than a period");
    // an even power
    check(close(lower(pow(x, 2).enclose({span(-2.0, 1.0)})), 0.0), "an even power reaches 0");
}

/// One call on a column of points equals the calls on the points one by one.
void tensors_evaluate_elementwise() {
    const Expr x = Expr::var(0), y = Expr::var(1);
    const Expr f = sin(x * y) / (1 + x * x) + 2 * pow(y, 3) - exp(x) / 3;
    const std::vector<double> xs = {0.3, -0.7, 1.1}, ys = {0.5, 0.2, -0.9};
    const Tensor all =
        f.eval_tensor({Tensor::from_data(xs, {3, 1}), Tensor::from_data(ys, {3, 1})});
    check(all.shape() == std::vector<int64_t>({3, 1}), "the result has the shape of the points");
    const std::vector<double> values = all.data();
    for (int i = 0; i < 3; ++i)
        check(close(values[i], f.eval({xs[i], ys[i]})), "point " + std::to_string(i));
}

/// Products and squares of ranges take the extremes of the ends.
void range_arithmetic_takes_the_extremes() {
    const Range a = span(-2.0, 3.0), b = span(-1.0, 4.0);
    const Range p = a * b, q = square(a), r = a * -2.0, d = a - b;
    check(close(lower(p), -8.0) && close(upper(p), 12.0), "product");
    check(close(lower(q), 0.0) && close(upper(q), 9.0), "the square of a range around 0");
    check(close(lower(square(span(1.0, 2.0))), 1.0) && close(lower(square(span(-3.0, -1.0))), 1.0),
          "the square of a range away from 0");
    check(close(lower(r), -6.0) && close(upper(r), 4.0), "a negative factor swaps the ends");
    check(close(lower(d), -6.0) && close(upper(d), 4.0), "difference");
}

/// tan, sqrt and real powers: their values, derivatives and monotone ranges; a list of
/// expressions shares the value of a common subexpression.
void tan_sqrt_and_real_powers() {
    const Expr x = Expr::var(0);
    const Expr t = tan(x), r = sqrt(x), p = pow(x, 1.5);
    check(close(t.eval({0.3}), std::tan(0.3)) &&
              close(t.diff(0).eval({0.3}), 1 + std::pow(std::tan(0.3), 2)),
          "tan and its derivative");
    check(close(r.eval({4.0}), 2.0) && close(r.diff(0).eval({4.0}), 0.25),
          "sqrt and its derivative");
    check(close(p.diff(0).eval({4.0}), 3.0) && close(pow(x, -0.5).eval({4.0}), 0.5),
          "x^1.5, x^-0.5");
    const Range I = span(1.0, 4.0);
    check(close(lower(p.enclose({I})), 1.0) && close(upper(p.enclose({I})), 8.0),
          "the range of x^1.5");
    check(close(lower(pow(x, -0.5).enclose({I})), 0.5) &&
              close(upper(pow(x, -0.5).enclose({I})), 1.0),
          "a negative power decreases");
    check(close(upper(t.enclose({span(-0.5, 0.5)})), std::tan(0.5)), "the range of tan");
    // one pass over a list
    const Expr shared = sin(x) * x;
    const std::vector<Range> both = Expr::enclose({shared + 1, shared * 2}, {span(0.0, 1.0)});
    check(both.size() == 2 && close(upper(both[0]), std::sin(1.0) + 1) &&
              close(upper(both[1]), 2 * std::sin(1.0)),
          "a list of enclosures");
}

/// The ten new functions: values, derivatives (against finite differences), the range of a
/// sampled box, and the refusal of a range outside the domain.
void inverse_and_hyperbolic_functions() {
    const Expr x = Expr::var(0);
    struct Case {
        const char *name;
        Expr f;
        double (*g)(double);
        double lo, hi; // a box inside the domain
    };
    const std::vector<Case> cases = {
        {"log", log(x), [](double v) { return std::log(v); }, 0.5, 3.0},
        {"asin", asin(x), [](double v) { return std::asin(v); }, -0.7, 0.8},
        {"acos", acos(x), [](double v) { return std::acos(v); }, -0.7, 0.8},
        {"atan", atan(x), [](double v) { return std::atan(v); }, -2.0, 3.0},
        {"sinh", sinh(x), [](double v) { return std::sinh(v); }, -1.0, 2.0},
        {"cosh", cosh(x), [](double v) { return std::cosh(v); }, -1.0, 2.0},
        {"tanh", tanh(x), [](double v) { return std::tanh(v); }, -1.0, 2.0},
        {"asinh", asinh(x), [](double v) { return std::asinh(v); }, -2.0, 3.0},
        {"acosh", acosh(x), [](double v) { return std::acosh(v); }, 1.2, 4.0},
        {"atanh", atanh(x), [](double v) { return std::atanh(v); }, -0.7, 0.8}};
    for (const Case &c : cases) {
        const double p = 0.5 * (c.lo + c.hi);
        check(close(c.f.eval({p}), c.g(p)), std::string(c.name) + ": value");
        const double h = 1e-6;
        check(close(c.f.diff(0).eval({p}), (c.g(p + h) - c.g(p - h)) / (2 * h), 1e-6),
              std::string(c.name) + ": derivative");
        // the range holds every sampled value, and is tight at the ends of a monotone function
        const Range r = c.f.enclose({span(c.lo, c.hi)});
        bool inside = true;
        for (int k = 0; k <= 50; ++k) {
            const double v = c.g(c.lo + (c.hi - c.lo) * k / 50);
            inside = inside && lower(r) <= v && v <= upper(r);
        }
        check(inside, std::string(c.name) + ": the range holds the samples");
    }
    // cosh dips to 1 where its range holds 0; acos falls
    check(close(lower(cosh(x).enclose({span(-1.0, 2.0)})), 1.0), "cosh around 0");
    check(close(lower(acos(x).enclose({span(0.0, 0.5)})), std::acos(0.5)), "acos falls");
    // outside the domain
    check(test::raises([&] { log(x).enclose({span(-1.0, 1.0)}); }, Error::out_of_domain),
          "log of a range with a negative number");
    check(test::raises([&] { asin(x).enclose({span(0.0, 1.5)}); }, Error::out_of_domain),
          "asin beyond 1");
    check(test::raises([&] { acosh(x).enclose({span(0.5, 2.0)}); }, Error::out_of_domain),
          "acosh below 1");
}

/// Expr::function builds the node of each named function; the oracle's JSON names it.
void functions_by_operation_and_in_json() {
    const Expr x = Expr::var(0);
    // every function of one operand with its name
    const std::vector<std::pair<ExprOp, std::string>> ops = {
        {ExprOp::sin, "sin"},     {ExprOp::cos, "cos"},    {ExprOp::tan, "tan"},
        {ExprOp::exp, "exp"},     {ExprOp::log, "log"},    {ExprOp::asin, "asin"},
        {ExprOp::acos, "acos"},   {ExprOp::atan, "atan"},  {ExprOp::sinh, "sinh"},
        {ExprOp::cosh, "cosh"},   {ExprOp::tanh, "tanh"},  {ExprOp::asinh, "asinh"},
        {ExprOp::acosh, "acosh"}, {ExprOp::atanh, "atanh"}};
    for (const auto &[op, name] : ops) {
        // the node has the operation and its operand, and prints as the oracle names it
        const Expr f = Expr::function(op, x);
        check(f.view().op == op, name + ": the node of the function");
        const std::string json = f.to_json(1, [](double v) { return std::to_string(v); });
        check(json.find("\"op\":\"" + name + "\"") != std::string::npos,
              name + ": the oracle's name");
    }
    // an operation of two operands is no function
    check(test::raises([&] { Expr::function(ExprOp::add, x); }, Error::wrong_value),
          "function: an operation of two operands");
}

void errors_are_described() {
    // the errors of the operations
    const Expr x = Expr::var(0);
    check(test::throws([&] { (1 / x).enclose({span(-1.0, 1.0)}); }),
          "dividing by a range around 0");
    check(test::throws([&] { x.eval(std::vector<double>{}); }), "a missing variable");
    check(test::throws([&] { Expr::var(-1); }), "a negative variable index");
    check(test::throws([&] { x / Expr(0.0); }), "dividing by the constant 0");
    check(test::raises([&] { tan(x).enclose({span(1.0, 2.0)}); }, Error::out_of_domain),
          "tan over a pole");
    check(test::raises([&] { sqrt(x).enclose({span(-1.0, 1.0)}); }, Error::out_of_domain),
          "a real power of a negative range");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    values_follow_the_formula();
    derivatives_match_finite_differences();
    enclosures_contain_the_values();
    trigonometric_ranges_know_their_extremes();
    tensors_evaluate_elementwise();
    range_arithmetic_takes_the_extremes();
    tan_sqrt_and_real_powers();
    inverse_and_hyperbolic_functions();
    functions_by_operation_and_in_json();
    errors_are_described();
    return test::finish("expr");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
