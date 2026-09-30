// test_expr - symbolic expressions: values, derivatives, and enclosures over boxes

#include "global/expr.h"
#include "testing.h"

#include <cmath>
#include <functional>

using namespace cora;
using test::check;
using test::close;

namespace {

/// The range [lo, hi] of one number, and the ends of a range of one number.
Range span(double lo, double hi) {
    return Range(Tensor::fromData({lo}, {1, 1}), Tensor::fromData({hi}, {1, 1}));
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
                check(v >= lower(r) - 1e-12 && v <= upper(r) + 1e-12, "a sampled value is enclosed");
            }
    }
}

void trigonometric_ranges_know_their_extremes() {
    const Expr x = Expr::var(0);
    const Range s = sin(x).enclose({span(0.0, 3.0)}), c = cos(x).enclose({span(-1.0, 1.0)});
    check(close(upper(s), 1.0) && close(lower(s), 0.0), "sin over [0, 3] peaks at 1");
    check(close(upper(c), 1.0) && close(lower(c), std::cos(1.0)), "cos over [-1, 1] peaks at 0");
    check(close(lower(sin(x).enclose({span(0.0, 7.0)})), -1.0), "a range of more than a period");
    check(close(lower(pow(x, 2).enclose({span(-2.0, 1.0)})), 0.0), "an even power reaches 0");
}

/// One call on a column of points equals the calls on the points one by one.
void tensors_evaluate_elementwise() {
    const Expr x = Expr::var(0), y = Expr::var(1);
    const Expr f = sin(x * y) / (1 + x * x) + 2 * pow(y, 3) - exp(x) / 3;
    const std::vector<double> xs = {0.3, -0.7, 1.1}, ys = {0.5, 0.2, -0.9};
    const Tensor all = f.evalTensor({Tensor::fromData(xs, {3, 1}), Tensor::fromData(ys, {3, 1})});
    check(all.shape() == std::vector<int64_t>({3, 1}), "the result has the shape of the points");
    const std::vector<double> values = all.data();
    for (int i = 0; i < 3; ++i) check(close(values[i], f.eval({xs[i], ys[i]})), "point " + std::to_string(i));
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

void errors_are_described() {
    const Expr x = Expr::var(0);
    check(test::throws([&] { (1 / x).enclose({span(-1.0, 1.0)}); }), "dividing by a range around 0");
    check(test::throws([&] { x.eval(std::vector<double>{}); }), "a missing variable");
    check(test::throws([&] { Expr::var(-1); }), "a negative variable index");
    check(test::throws([&] { x / Expr(0.0); }), "dividing by the constant 0");
}

} // namespace

int main() {
    values_follow_the_formula();
    derivatives_match_finite_differences();
    enclosures_contain_the_values();
    trigonometric_ranges_know_their_extremes();
    tensors_evaluate_elementwise();
    range_arithmetic_takes_the_extremes();
    errors_are_described();
    return test::finish("expr");
}
