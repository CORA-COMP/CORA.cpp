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
# The Python module is a shared object, so every object it links must be relocatable.
PIC        = -fPIC
PYTHON    ?= python3
CXXFLAGS  ?= -O3 -march=native -std=c++17 -fopenmp -DNDEBUG -DEIGEN_NO_DEBUG
INCLUDES   = -Isrc -Icompetition -isystem $(EIGEN)
LDLIBS     = -lglpk
LDFLAGS    = -fopenmp

# The library: src/. The competition's harness around it: competition/.
SRC = src/rng.cpp src/threads.cpp src/contSet/sets.cpp src/contSet/contains.cpp \
      src/contSet/lp.cpp src/contSet/zonotope.cpp src/tensor/tensor.cpp \
      src/tensor/eigen.cpp src/contDynamics/linear_sys.cpp \
      competition/json.cpp competition/catalog.cpp competition/instance.cpp \
      competition/server.cpp

ifeq ($(TORCH),)
SRC += competition/torch_none.cpp
else
SRC      += competition/torch_backend.cpp src/contSet/torch_contains.cpp src/tensor/torch.cpp
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

OBJ = $(SRC:%.cpp=build/%.o)

all: build/coracpp

build/coracpp: $(OBJ) build/competition/main.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

build/src/rng.o: src/rng.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) -ffast-math $(WARN) $(INCLUDES) -c -o $@ $<

build/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -c -o $@ $<

TESTS = build/test_ops build/test_linear_sys
ifneq ($(TORCH),)
TESTS += build/test_torch build/test_linear_sys_torch
endif

build/test_%: tests/test_%.cpp $(OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEFS) $(WARN) $(INCLUDES) -Itests $(LDFLAGS) -o $@ $^ $(LDLIBS)

build/example_%: examples/%.cpp $(OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LDLIBS)

example: build/example_linear_sys

test: $(TESTS)
	@for t in $(TESTS); do echo "== $$t"; ./$$t || exit 1; done

# `make python TORCH=...` builds the `coracpp` module into build/; put build/ on PYTHONPATH.
# It needs libtorch, since torch tensors are its inputs.
PYEXT = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYINC = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")

python: build/coracpp$(PYEXT)

build/coracpp$(PYEXT): src/python/bindings.cpp $(OBJ)
	@test -n "$(TORCH)" || { echo "make python needs TORCH=/path/to/libtorch"; exit 1; }
	$(CXX) -shared $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -I$(PYINC) $(LDFLAGS) \
	    -o $@ $^ $(LDLIBS) -L$(TORCH)/lib -ltorch_python

clean:
	rm -rf build

.PHONY: all test clean python example
