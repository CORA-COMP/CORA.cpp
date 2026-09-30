// test_linearSys_verify_stub - verify is declared and links; until the algorithms land it throws

#include "contDynamics/linearSys/linearSys.h"
#include "testing.h"

#include <stdexcept>

using namespace cora;

int main() {
    const LinearSys sys(Tensor({{-1.0}}), Tensor({{1.0}}));
    const Zonotope R0(Tensor({1.0}), Tensor({{0.1}})), U(Tensor({0.0}), Tensor({{0.1}}));
    bool notImplemented = false;
    try {
        sys.verify({R0, U, 1.0}, VerifyAlg::Zonotope, {});
    } catch (const std::logic_error &) {
        notImplemented = true;
    }
    test::check(notImplemented, "verify: the stub throws std::logic_error");
    return test::finish("linearSys verify (stub)");
}
