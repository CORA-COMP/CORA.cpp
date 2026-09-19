# CORA.cpp — one binary, no build system beyond this file.
#
# Eigen is header-only and GLPK is the only library; -march=native is safe because the
# tool is built on the worker it runs on. rng.cpp is the one file compiled with
# -ffast-math, so the compiler may vectorize the logarithm and the sine of Box–Muller
# through libmvec; everything that the results depend on keeps strict IEEE arithmetic.

CXX      ?= g++
EIGEN    ?= /usr/include/eigen3
WARN      = -Wall -Wextra
CXXFLAGS ?= -O3 -march=native -std=c++17 -fopenmp -DNDEBUG -DEIGEN_NO_DEBUG
INCLUDES  = -Isrc -isystem $(EIGEN)
LDLIBS    = -lglpk
LDFLAGS   = -fopenmp

SRC  = src/rng.cpp src/json.cpp src/sets.cpp src/contains.cpp src/lp.cpp \
       src/instance.cpp src/server.cpp
OBJ  = $(SRC:src/%.cpp=build/%.o)

all: build/coracpp

build/coracpp: $(OBJ) build/main.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

build/rng.o: src/rng.cpp | build
	$(CXX) $(CXXFLAGS) -ffast-math $(WARN) $(INCLUDES) -c -o $@ $<

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) $(WARN) $(INCLUDES) -c -o $@ $<

build/test_ops: tests/test_ops.cpp $(OBJ) | build
	$(CXX) $(CXXFLAGS) $(WARN) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LDLIBS)

test: build/test_ops
	./build/test_ops

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all test clean
