// test_expr - symbolic expressions: values, derivatives, and enclosures over boxes

#include "global/expr.h"
#include "testing.h"

#include <cmath>
#include <functional>

using namespace cora::ct;
using test::check;
using test::close;

namespace {

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
    const std::vector<Range> box = {{-1.2, 2.0}, {-3.0, 1.5}};
    for (const Expr &f : fs) {
        const Range r = f.enclose(box);
        for (int i = 0; i <= 20; ++i)
            for (int j = 0; j <= 20; ++j) {
                const double v = f.eval({-1.2 + 3.2 * i / 20, -3.0 + 4.5 * j / 20});
                check(v >= r.lo - 1e-12 && v <= r.hi + 1e-12, "a sampled value is enclosed");
            }
    }
}

void trigonometric_ranges_know_their_extremes() {
    const Expr x = Expr::var(0);
    const Range s = sin(x).enclose({{0.0, 3.0}}), c = cos(x).enclose({{-1.0, 1.0}});
    check(close(s.hi, 1.0) && close(s.lo, 0.0), "sin over [0, 3] peaks at 1");
    check(close(c.hi, 1.0) && close(c.lo, std::cos(1.0)), "cos over [-1, 1] peaks at 0");
    check(close(sin(x).enclose({{0.0, 7.0}}).lo, -1.0), "a range of more than a period");
    check(close(pow(x, 2).enclose({{-2.0, 1.0}}).lo, 0.0), "an even power reaches 0");
}

void errors_are_described() {
    const Expr x = Expr::var(0);
    check(test::throws([&] { (1 / x).enclose({{-1.0, 1.0}}); }), "dividing by a range around 0");
    check(test::throws([&] { x.eval({}); }), "a missing variable");
    check(test::throws([&] { Expr::var(-1); }), "a negative variable index");
    check(test::throws([&] { x / Expr(0.0); }), "dividing by the constant 0");
}

} // namespace

int main() {
    values_follow_the_formula();
    derivatives_match_finite_differences();
    enclosures_contain_the_values();
    trigonometric_ranges_know_their_extremes();
    errors_are_described();
    return test::finish("expr");
}
