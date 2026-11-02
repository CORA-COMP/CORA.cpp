// test_linear_sys_expm_remainder - the error box of the truncated exponential series, a public
// building block of reach

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// The box is symmetric around 0 and shrinks with more Taylor terms.
void the_box_shrinks_with_the_terms() {
    const LinearSys sys = LinearSys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}));
    const Interval E4 = sys.expm_remainder(0.1, 4), E8 = sys.expm_remainder(0.1, 8);
    check(test::close(E4.inf() + E4.sup(), std::vector<double>(4, 0.0), 1e-15), "symmetric");
    bool smaller = true;
    const std::vector<double> w4 = E4.sup().data(), w8 = E8.sup().data();
    for (std::size_t i = 0; i < w4.size(); ++i) smaller = smaller && w8[i] <= w4[i] && w8[i] >= 0;
    check(smaller, "8 terms leave a smaller box than 4");
}

/// A step size of zero and no terms are refused; the checks off give the same box.
void checks_and_their_switch() {
    const LinearSys sys = LinearSys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}));
    check(test::raises([&] { sys.expm_remainder(0.0, 4); }, Error::wrong_value), "time_step 0");
    check(test::raises([&] { sys.expm_remainder(0.1, 0); }, Error::wrong_value), "taylor_terms 0");
    const Tensor on = sys.expm_remainder(0.1, 4).sup();
    ChecksScope off(false);
    check(test::close(sys.expm_remainder(0.1, 4).sup(), on, 0), "the same with the checks off");
}

} // namespace

int main() {
    the_box_shrinks_with_the_terms();
    checks_and_their_switch();
    return test::finish("linear_sys expm_remainder");
}
