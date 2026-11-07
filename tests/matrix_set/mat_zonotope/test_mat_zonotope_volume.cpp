// test_mat_zonotope_volume - MatZonotope::volume: the volume of a matrix zonotope
//
// CORA: matrixSet/@matZonotope/volume.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // two generators e_1 and 2 e_2 of the stacked plane: the volume is 2^2 * 2
        const MatZonotope M(test::matrix({0, 0}, 1, 2), Tensor::from_data({1, 0, 0, 2}, {2, 1, 2}));
        check(close(M.volume().item(), 8.0), b + ": 2^n |det G|");
        check(close(M.volume().item(), M.zonotope().volume().item(), 0),
              b + ": the volume of the stacked zonotope");
        check(close(test::example().volume().item(), 0.0), b + ": generators that do not span: 0");
    });
    check(test::raises([] { test::example().volume("nonsense"); }, Error::wrong_value),
          "volume: a method");
    ChecksScope off(false);
    check(close(test::example().volume().item(), 0.0), "volume: the same with the checks off");
    return test::finish("mat_zonotope volume");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
