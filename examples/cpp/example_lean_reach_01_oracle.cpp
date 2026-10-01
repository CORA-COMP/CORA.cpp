// example_lean_reach_01_oracle - reachable sets with CORALean as the oracle, in floating point
//
// CORALean computes zonotopes whose float operations are sound: a nominal zonotope plus an error
// box that encloses every rounding error. Here a damped oscillator is propagated in binary64 and
// in binary32, the nominal part and the error box are taken apart, and both are compared with
// CORA.cpp's own reachable set. A value crosses back into CORA.cpp only when it is exact.
//
// Syntax:   CORACPP_ORACLE="cd CORALean && lake exe oracle" build/examples/cpp/example_lean_reach_01_oracle
// Outputs:  per dtype the final interval of the nominal part, the width of the error box, and
//           whether the lean set encloses CORA.cpp's; without an oracle a note and exit code 0
// See also: example_linear_reach_01_5dim, lean/zonotope.h

#include "contDynamics/linearSys/linearSys.h"
#include "global/oracle/linearSys.h"
#include "global/oracle/oracle.h"

#include <cstdlib>
#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

int main() {
    if (!std::getenv("CORACPP_ORACLE")) {
        std::cout << "set CORACPP_ORACLE to the command that starts the CORALean oracle\n";
        return 0;
    }

    // Parameters ------------------------------------------------------------------------------

    // every value is a dyadic number, so it is exact in binary64 and in binary32
    const Tensor A({{0.0, 1.0}, {-1.0, -0.25}});
    const Tensor c({{1.0}, {0.0}}), G({{0.25, 0.0}, {0.0, 0.25}});
    const double timeStep = 0.125, tFinal = 2.0;  // a power of two: exact in every dtype
    const int taylorTerms = 8, zonotopeOrder = 10;

    // CORA.cpp's own result, in double without a rounding-error account
    const Reach ref = LinearSys(A).reach(Zonotope(c, G), timeStep, tFinal, taylorTerms);
    const Interval last = ref.timePoint.back().interval();
    std::cout << "CORA.cpp:  x1 in [" << last.inf.data()[0] << ", " << last.sup.data()[0] << "]\n";

    // The Oracle ------------------------------------------------------------------------------

    for (const std::string dtype : {"binary64", "ieee:binary32"}) {
        lean::setDType(dtype);  // the lean objects below are created in this dtype
        const lean::ReachSet R = lean::LinearSys(A).reach(lean::Zonotope(c, G), timeStep, tFinal,
                                                          taylorTerms, zonotopeOrder);

        // the nominal zonotope and the error box that the sound float operations added
        const lean::Zonotope &Z = R.timePoint.back();
        const auto [nominal, error] = Z.gather();
        const Interval hull = nominal.interval();
        std::cout << dtype << ": x1 in [" << hull.inf.data()[0] << ", " << hull.sup.data()[0]
                  << "], error box radius " << error.sup.data()[0] << "\n";

        // the enclosure (nominal plus error) must contain CORA.cpp's interval at every time
        bool encloses = true;
        for (std::size_t k = 0; k < R.timePoint.size(); ++k) {
            const auto [lo, hi] = R.timePoint[k].interval().gather();
            const Interval I = ref.timePoint[k].interval();
            for (int64_t i = 0; i < 2; ++i)
                encloses = encloses && lo.data()[i] <= I.inf.data()[i] + 1e-9 &&
                           hi.data()[i] >= I.sup.data()[i] - 1e-9;
        }
        std::cout << "  encloses the CORA.cpp sets at all " << R.timePoint.size()
                  << " time points: " << (encloses ? "yes" : "NO") << "\n";
    }

    // Rounding --------------------------------------------------------------------------------

    // 0.1 is not a binary32 number: roundTo returns the rounded value and an enclosure of the error
    lean::setDType("binary64");
    const auto [value, error] = lean::Tensor::from(Tensor({{0.1}})).roundTo("ieee:binary32");
    const auto [inf, sup] = error.gather();
    std::cout << "0.1 in binary32 is " << value.gather().data()[0] << ", the error lies in ["
              << inf.data()[0] << ", " << sup.data()[0] << "]\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
