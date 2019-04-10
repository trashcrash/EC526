.SUFFIXES:
.SUFFIXES: .o .cpp

COMPILER = GCC
#COMPILER = PGI
#COMPILER = PGI-tesla
#COMPILER = PGI-multicore

CXX=pgc++
ifeq ($(COMPILER),GCC)
	CXX=g++
	CXXFLAGS += -std=c++11
else ifeq ($(COMPILER),PGI)
	CXXFLAGS += -fast
else ifeq ($(COMPILER),PGI-tesla)
	CXXFLAGS += -Minfo=accel -fast -acc -ta=tesla:cc60
else ifeq ($(COMPILER),PGI-multicore)
	CXXFLAGS += -Minfo=accel -fast -acc -ta=multicore
endif

#============================================================
TARGET = phi

C_SOURCES = phi.cpp
C_OBJS    = phi.o

#============================================================
all: $(TARGET)

.o:.cpp
	$(CXX) -c $(CXXFLAGS) $<  

$(TARGET) :   $(C_OBJS)
	$(CXX) $(CXXFLAGS)  $^ $(LIBDIRS)  -o $@

# Implicit rules: $@ = target name, $< = first prerequisite name, $^ = name of all prerequisites 
#============================================================

ALL_SOURCES = Makefile $(C_SOURCES)

clean:
	rm -f $(TARGET) $(C_OBJS) *~*~ core
