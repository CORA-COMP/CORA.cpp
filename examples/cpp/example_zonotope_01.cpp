// example_zonotope_01 - a zonotope that is rotated and shrunk again and again
//
// Operators as in CORA: A * Z maps the zonotope, and std::cout << Z prints it. A plain plot(Z)
// takes the next color of CORA's order, so every copy has its own (blue, red, yellow, ...).
//
// Syntax:   build/examples/cpp/example_zonotope_01
// Outputs:  the matrix and the zonotope as text, and the figure as an SVG file
// See also: example_zonotope_01 of the Python examples, example_linear_reach_01_5dim

#include "global/plot/plot.h"

#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora::ct;

int main() {
    // Init ------------------------------------------------------------------------------------

    // init zonotope
    Zonotope Z(Tensor({-1, 0}), Tensor({{1, 0}, {0, 1}}) * 0.1);

    // init rotation matrix
    const double phi = 0.2;  // radians
    Tensor A({{Tensor::cos(phi), Tensor::sin(phi)}, {-Tensor::sin(phi), Tensor::cos(phi)}});
    A = A * 0.975;

    // display
    std::cout << A;
    std::cout << Z;

    // Visualization ---------------------------------------------------------------------------

    for (int i = 0; i < 100; ++i) {
        plot(Z);
        Z = A * Z;
    }

    // save figure
    figure().title = "zonotope";
    figure().save("example_zonotope_01.svg");

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
