// test_lean_mat_zonotope - the operations of MatZonotope against the CORALean oracle (its
// `matZonotope.*` ops)
//
// At nearest the nominal part (the center and the generators, in their order) is the oracle's bit for
// bit; in the sound mode the box of CORA.cpp's set lies in the box of the oracle's enclosure.
//
// ORACLE: matrixSet/@matZonotope/center.m
// ORACLE: matrixSet/@matZonotope/intervalMatrix.m
// ORACLE: matrixSet/@matZonotope/mpower.m
// ORACLE: matrixSet/@matZonotope/mtimes.m
// ORACLE: matrixSet/@matZonotope/plus.m
// ORACLE: matrixSet/@matZonotope/powers.m
// ORACLE: matrixSet/@matZonotope/representsa.m
// ORACLE: matrixSet/@matZonotope/transpose.m
// ORACLE: matrixSet/@matZonotope/zonotope.m

#include "cora/global/oracle/mat_zonotope.h"
#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "oracle_testing.h"

using namespace cora;
using test::oracle::check;
using test::oracle::Random;

int main() {
    if (!test::oracle::available()) return 0;
    // (rows, cols, generators)
    const std::vector<std::array<int64_t, 3>> shapes{{2, 2, 1}, {3, 2, 2}, {4, 4, 3}};
    test::oracle::run("matZonotope.center", [&](Random &rnd) {
        for (const auto &[r, c, h] : shapes) {
            const MatZonotope A = rnd.mat_zonotope(r, c, h);
            check(A.center(), lean::MatZonotope(A).center_value());
        }
    });
    test::oracle::run("matZonotope.intervalMatrix", [&](Random &rnd) {
        for (const auto &[r, c, h] : shapes) {
            const MatZonotope A = rnd.mat_zonotope(r, c, h);
            check(A.interval_matrix(), lean::MatZonotope(A).interval_matrix());
        }
    });
    test::oracle::run("matZonotope.transpose", [&](Random &rnd) {
        for (const auto &[r, c, h] : shapes) {
            const MatZonotope A = rnd.mat_zonotope(r, c, h);
            check(A.transpose(), lean::MatZonotope(A).transpose());
        }
    });
    test::oracle::run("matZonotope.plus", [&](Random &rnd) {
        for (const auto &[r, c, h] : shapes) {
            const MatZonotope A = rnd.mat_zonotope(r, c, h), B = rnd.mat_zonotope(r, c, h + 1);
            check(A.plus(B), lean::MatZonotope(A).plus(lean::MatZonotope(B)));
        }
    });
    test::oracle::run("matZonotope.mtimes", [&](Random &rnd) {
        for (const auto &[r, c, h] : shapes) {
            const MatZonotope A = rnd.mat_zonotope(r, c, h), B = rnd.mat_zonotope(c, r + 1, h + 1);
            check(A.matmul(B), lean::MatZonotope(A).matmul(lean::MatZonotope(B)));
        }
    });
    test::oracle::run("matZonotope.mpower", [&](Random &rnd) {
        for (const auto &[n, unused, h] : shapes) {
            const MatZonotope A = rnd.mat_zonotope(n, n, std::min<int64_t>(h, 2));
            for (const int k : {0, 1, 2, 3})
                check(A.matrix_power(k), lean::MatZonotope(A).matrix_power(k));
        }
    });
    // CORA.cpp lists the powers P, P A, P A^2 (orders 1 to 3, the first being P); the oracle gives the
    // one at offset k
    test::oracle::run("matZonotope.powers", [&](Random &rnd) {
        for (const int64_t n : {2, 3}) {
            const MatZonotope A = rnd.mat_zonotope(n, n, 2), P = rnd.mat_zonotope(n, n, 1);
            const std::vector<MatZonotope> powers = A.powers(3, 1, P);
            for (int k = 0; k < 3; ++k)
                check(powers[k], lean::MatZonotope(A).powers(lean::MatZonotope(P), k));
        }
    });
    test::oracle::run("matZonotope.zonotope", [&](Random &rnd) {
        for (const auto &[r, c, h] : shapes) {
            const MatZonotope A = rnd.mat_zonotope(r, c, h);
            check(A.zonotope(), lean::MatZonotope(A).zonotope());
        }
    });
    test::oracle::run("matZonotope.representsa", [&](Random &rnd) {
        for (const auto &[r, c, h] : shapes) {
            const Tensor zeros = Tensor::zeros({h, r, c}), C = rnd.matrix(r, c);
            const MatZonotope random = rnd.mat_zonotope(r, c, h), point(C, zeros),
                              origin(C.zeros_like(), zeros);
            for (const char *type : {"point", "origin"})
                for (const MatZonotope &A : {random, point, origin})
                    check(A.representsa(type, 0.0), lean::MatZonotope(A).representsa(type));
        }
    });
    return test::finish("lean mat_zonotope");
}
