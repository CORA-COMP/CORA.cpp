// test_lean_tensor - lean::Tensor, Interval and Zonotope: the exact crossing and the dtype rules
//
// The parts that need no oracle run always; with CORACPP_ORACLE set, roundTo and the zonotope
// operations are checked against the oracle too.

#include "lean/oracle.h"
#include "lean/zonotope.h"
#include "testing.h"

#include <cstdlib>
#include <stdexcept>

using namespace cora;
using test::throws;

static lean::Json matrix(int64_t rows, int64_t cols, const std::vector<std::string> &values) {
    lean::Json j = lean::Json::object(), data = lean::Json::array();
    for (const std::string &v : values) data.push(v);
    j.set("rows", rows).set("cols", cols).set("data", data);
    return j;
}

int main() {
    test::for_each_backend([](const std::string &) {
        lean::setDType("binary64");

        // a double crosses exactly in both directions
        const Tensor x({{0.1, -2.5}, {1e-300, 3.0}});
        test::check(test::close(lean::Tensor::from(x).gather(), x, 0), "gather(from(x)) == x");

        // a value that is not a double is not gathered
        const lean::Tensor wide = lean::Tensor::fromJson("dyadic:60", matrix(1, 1, {"9007199254740993*2^0"}));
        test::check(throws([&] { wide.gather(); }), "a 54-bit value throws");
        const lean::Tensor fine = lean::Tensor::fromJson("dyadic:8", matrix(1, 1, {"3*2^-2"}));
        test::check(test::close(fine.gather(), std::vector<double>{0.75}, 0), "an exact dyadic gathers");

        // a fixedpoint decimal gathers only if it is also a double
        const lean::Tensor half = lean::Tensor::fromJson("fixedpoint:2", matrix(1, 1, {"5*10^-1"}));
        const lean::Tensor cent = lean::Tensor::fromJson("fixedpoint:2", matrix(1, 1, {"1*10^-2"}));
        test::check(test::close(half.gather(), std::vector<double>{0.5}, 0), "5*10^-1 is 0.5");
        test::check(throws([&] { cent.gather(); }), "0.01 is no double");

        // NaN and a batch do not enter
        test::check(throws([] { lean::Tensor::from(Tensor({{std::nan("")}})); }), "NaN throws");
        test::check(throws([] { lean::Tensor::from(Tensor::zeros({2, 2, 2})); }), "a batch throws");

        // unknown dtypes and rounding modes are errors
        test::check(throws([] { lean::setDType("float16"); }), "an unknown dtype throws");
        test::check(throws([&] { fine.roundTo("binary64", "sideways"); }), "an unknown mode throws");

        // two dtypes do not mix
        const lean::Tensor a = lean::Tensor::from(Tensor({{1.0}, {2.0}}));
        const lean::Interval box(a, a);
        const lean::Interval dyadic(fine, fine);
        test::check(throws([&] { lean::Zonotope(a, a, dyadic); }), "a zonotope of mixed dtypes throws");
        test::check(throws([&] { lean::Zonotope(a, a, box).plus(dyadic); }), "plus of mixed throws");
        test::check(throws([&] { lean::Zonotope(a, a, box).reduce(1, "pca"); }), "unknown method");

        if (!std::getenv("CORACPP_ORACLE")) return;

        // roundTo: 0.1 into dyadic:3 (3 significand bits: 0.09375); the error encloses x - value
        const lean::Tensor tenth = lean::Tensor::from(Tensor({{0.1}}));
        const auto [value, error] = tenth.roundTo("dyadic:3");
        const auto [inf, sup] = error.gather();
        test::check(test::close(value.gather(), std::vector<double>{0.09375}, 0), "0.1 rounds to 3*2^-5");
        const double gap = 0.1 - 0.09375;
        test::check(inf.data()[0] <= gap && gap <= sup.data()[0], "the error encloses");

        // the zonotope operations enclose the exact result
        const lean::Zonotope Z(Tensor({{1.0}, {0.0}}), Tensor({{0.5, 0.0}, {0.0, 0.25}}));
        const lean::Zonotope Zm = Z.mtimes(lean::Tensor::from(Tensor({{0.1, 0.0}, {0.0, 0.1}})));
        const auto [nominal, err] = Zm.gather();
        test::check(test::close(nominal.c, std::vector<double>{0.1, 0.0}, 1e-15), "nominal center");
        test::check(err.inf.data()[0] <= 0 && err.sup.data()[0] >= 0, "the error box contains 0");
    });
    return test::finish("lean tensor");
}
