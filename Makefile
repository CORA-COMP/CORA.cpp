# CORA.cpp - no build system beyond this file.
#
#   make test example                    build and run the tests and the C++ examples
#   make TORCH=/path/to/libtorch test    the same with libtorch (GPU, batching, gradients)
#   make python TORCH=/path/to/libtorch  the Python package, into build/coracpp
#   make all                             the CORA-COMP binary (see competition/README.md)
#   make test-competition                the competition's tests
#
# The library needs only Eigen (and libtorch for the parts that name it). The competition binary
# also needs GLPK. TORCH_ABI must be the _GLIBCXX_USE_CXX11_ABI that libtorch was built with;
# competition/install_tool.sh reads it off the install.

CXX       ?= g++
EIGEN     ?= /usr/include/eigen3
BLAS      ?=
TORCH     ?=
TORCH_ABI ?= 1
PYTHON    ?= python3
WARN       = -Wall -Wextra
PIC        = -fPIC  # the Python module is a shared object, so every object must be relocatable
CXXFLAGS  ?= -O3 -march=native -std=c++17 -fopenmp -DNDEBUG -DEIGEN_NO_DEBUG
INCLUDES   = -Isrc -isystem $(EIGEN)
LDLIBS     =
LDFLAGS    = -fopenmp

# ---------------------------------- the library ---------------------------------------
# src/ mirrors MATLAB CORA's folders, one operation per file; sources are found, so a new
# operation needs no edit here.

LIB_SRC = $(filter-out src/python/% src/tensor/torch.cpp,$(shell find src -name '*.cpp' | sort))

ifneq ($(TORCH),)
LIB_SRC  += src/tensor/torch.cpp
DEFS      = -DCORACPP_TORCH -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)
INCLUDES += -isystem $(TORCH)/include -isystem $(TORCH)/include/torch/csrc/api/include
LDFLAGS  += -L$(TORCH)/lib -Wl,-rpath,$(TORCH)/lib
LDLIBS   += -ltorch -ltorch_cpu -lc10
# --no-as-needed: nothing references libtorch_cuda directly, but without it linked the CUDA
# kernels are not registered and a GPU is never used.
ifneq ($(wildcard $(TORCH)/lib/libtorch_cuda.so),)
LDLIBS   += -Wl,--no-as-needed -ltorch_cuda -lc10_cuda -Wl,--as-needed
endif
endif

# An external BLAS is opt-in, since it changes which library does the arithmetic.
ifneq ($(BLAS),)
DEFS     += -DEIGEN_USE_BLAS
LDLIBS   += -lopenblas
endif

LIB_OBJ = $(LIB_SRC:%.cpp=build/%.o)

# rng.cpp is the one file compiled with -ffast-math, so the logarithm and the sine of
# Box-Muller vectorize; everything that results depend on keeps strict IEEE arithmetic.
build/src/global/rng.o: src/global/rng.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) -ffast-math $(WARN) $(INCLUDES) -c -o $@ $<

build/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -c -o $@ $<

# ---------------------------------- tests ----------------------------------------------
# tests/ mirrors src/; a test ending in _torch needs libtorch.

TEST_SRC = $(shell find tests -name 'test_*.cpp' -not -path 'tests/competition/*' | sort)
TESTS    = $(patsubst %.cpp,build/%,$(filter-out %_torch.cpp,$(TEST_SRC)))
ifneq ($(TORCH),)
TESTS   += $(patsubst %.cpp,build/%,$(filter %_torch.cpp,$(TEST_SRC)))
endif

$(TESTS): build/tests/%: tests/%.cpp $(LIB_OBJ) tests/testing.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -Itests $(LDFLAGS) -o $@ $(filter %.cpp %.o,$^) $(LDLIBS)

test: $(TESTS)
	@for t in $(TESTS); do echo "== $$t"; ./$$t || exit 1; done

# --------------------------------- examples --------------------------------------------
# One small file per topic in examples/cpp; the ones that name libtorch types need it.

EXAMPLE_SRC = $(sort $(wildcard examples/cpp/example_*.cpp))
EXAMPLES    = $(patsubst %.cpp,build/%,$(filter-out %_torch.cpp %_gpu.cpp %_batch.cpp %_gradient.cpp,$(EXAMPLE_SRC)))
ifneq ($(TORCH),)
EXAMPLES    = $(patsubst %.cpp,build/%,$(EXAMPLE_SRC))
endif

build/examples/cpp/%: examples/cpp/%.cpp $(LIB_OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LDLIBS)

example: $(EXAMPLES)
	@for e in $(EXAMPLES); do echo "== $$e"; ./$$e || exit 1; done

# ---------------------------------- Python ---------------------------------------------
# The coracpp package: the compiled module and the Python files beside it, in build/coracpp.
# Put build/ on PYTHONPATH. It needs libtorch, since torch tensors are its inputs.

PYEXT   = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYINC   = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYFILES = $(wildcard src/python/coracpp/*.py)

python: build/coracpp/_coracpp$(PYEXT) $(PYFILES:src/python/coracpp/%=build/coracpp/%)

build/coracpp/%.py: src/python/coracpp/%.py
	@mkdir -p $(@D)
	cp $< $@

build/coracpp/_coracpp$(PYEXT): src/python/bindings.cpp $(LIB_OBJ)
	@test -n "$(TORCH)" || { echo "make python needs TORCH=/path/to/libtorch"; exit 1; }
	@mkdir -p $(@D)
	$(CXX) -shared $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -I$(PYINC) $(LDFLAGS) \
	    -o $@ $^ $(LDLIBS) -L$(TORCH)/lib -ltorch_python

# ------------------------------ the competition ----------------------------------------
# competition/ is the CORA-COMP harness and its own tuned sets; it builds on the library and
# needs GLPK. See competition/README.md.

COMP_SRC = $(filter-out competition/main.cpp competition/torch_backend.cpp \
                        competition/torch_none.cpp competition/sets/torch_contains.cpp, \
                        $(shell find competition -name '*.cpp' | sort))
ifeq ($(TORCH),)
COMP_SRC += competition/torch_none.cpp
else
COMP_SRC += competition/torch_backend.cpp competition/sets/torch_contains.cpp
endif
COMP_OBJ = $(COMP_SRC:%.cpp=build/%.o)
COMP_LIBS = -lglpk

COMP_TEST_SRC = $(shell find tests/competition -name 'test_*.cpp' | sort)
COMP_TESTS    = $(patsubst %.cpp,build/%,$(filter-out %_torch.cpp,$(COMP_TEST_SRC)))
ifneq ($(TORCH),)
COMP_TESTS   += $(patsubst %.cpp,build/%,$(filter %_torch.cpp,$(COMP_TEST_SRC)))
endif

all: build/coracpp

build/coracpp: $(LIB_OBJ) $(COMP_OBJ) build/competition/main.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS) $(COMP_LIBS)

test-competition: $(COMP_TESTS)
	@for t in $(COMP_TESTS); do echo "== $$t"; ./$$t || exit 1; done

$(COMP_TESTS): build/tests/%: tests/%.cpp $(LIB_OBJ) $(COMP_OBJ) tests/testing.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(PIC) $(DEFS) $(WARN) $(INCLUDES) -Icompetition -Itests $(LDFLAGS) -o $@ $(filter %.cpp %.o,$^) $(LDLIBS) $(COMP_LIBS)

clean:
	rm -rf build

.PHONY: all test test-competition clean python example
