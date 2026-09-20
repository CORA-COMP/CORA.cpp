# CORA.cpp — one binary, no build system beyond this file.
#
# Eigen is header-only and GLPK is the only library the CPU path needs; -march=native is
# safe because the tool is built on the worker it runs on. rng.cpp is the one file
# compiled with -ffast-math, so the compiler may vectorize the logarithm and the sine of
# Box–Muller through libmvec; everything that the results depend on keeps strict IEEE
# arithmetic.
#
# `make TORCH=/path/to/libtorch` builds the libtorch backend in as well, which is what
# reaches the GPU and carries autograd; without it torch_none.cpp stands in and gpu
# instances report unsupported. TORCH_ABI must be the _GLIBCXX_USE_CXX11_ABI that
# libtorch itself was built with — install_tool.sh reads it off the install.

CXX       ?= g++
EIGEN     ?= /usr/include/eigen3
BLAS      ?=
TORCH     ?=
TORCH_ABI ?= 1
WARN       = -Wall -Wextra
CXXFLAGS  ?= -O3 -march=native -std=c++17 -fopenmp -DNDEBUG -DEIGEN_NO_DEBUG
INCLUDES   = -Isrc -isystem $(EIGEN)
LDLIBS     = -lglpk
LDFLAGS    = -fopenmp

SRC = src/rng.cpp src/json.cpp src/catalog.cpp src/threads.cpp src/sets.cpp \
      src/contains.cpp src/lp.cpp src/instance.cpp src/server.cpp

ifeq ($(TORCH),)
SRC += src/torch_none.cpp
else
SRC      += src/torch_backend.cpp src/torch_contains.cpp
DEFS      = -DCORACPP_TORCH -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)
INCLUDES += -isystem $(TORCH)/include -isystem $(TORCH)/include/torch/csrc/api/include
LDFLAGS  += -L$(TORCH)/lib -Wl,-rpath,$(TORCH)/lib
LDLIBS   += -ltorch -ltorch_cpu -lc10
# --no-as-needed: nothing references libtorch_cuda directly, but without it linked the
# CUDA kernels are not registered and every gpu instance would report unsupported.
ifneq ($(wildcard $(TORCH)/lib/libtorch_cuda.so),)
LDLIBS += -Wl,--no-as-needed -ltorch_cuda -lc10_cuda -Wl,--as-needed
endif
endif

# Eigen's own GEMM reaches about a fifth of this machine's peak; an external BLAS is
# what the catalog's largest matMul needs to finish at all. Opt-in, since it changes
# which library does the arithmetic.
ifneq ($(BLAS),)
DEFS   += -DEIGEN_USE_BLAS
LDLIBS += -lopenblas
endif

OBJ = $(SRC:src/%.cpp=build/%.o)

all: build/coracpp

build/coracpp: $(OBJ) build/main.o | build
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

build/rng.o: src/rng.cpp | build
	$(CXX) $(CXXFLAGS) $(DEFS) -ffast-math $(WARN) $(INCLUDES) -c -o $@ $<

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) $(DEFS) $(WARN) $(INCLUDES) -c -o $@ $<

TESTS = build/test_ops
ifneq ($(TORCH),)
TESTS += build/test_torch
endif

build/test_%: tests/test_%.cpp $(OBJ) | build
	$(CXX) $(CXXFLAGS) $(DEFS) $(WARN) $(INCLUDES) -Itests $(LDFLAGS) -o $@ $^ $(LDLIBS)

test: $(TESTS)
	@for t in $(TESTS); do echo "== $$t"; ./$$t || exit 1; done

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all test clean
