.SUFFIXES:
.SUFFIXES: .o .cpp

#============================================================
TARGET1 = time_dependent_v
TARGET2 = time_dependent_gauss
TARGET3 = time_dependent_w

C_SOURCES = time_dependent_v.cpp time_dependent_gauss.cpp time_dependent_w.cpp
C_OBJS1 = time_dependent_v.o
C_OBJS2 = time_dependent_gauss.o
C_OBJS3 = time_dependent_w.o
MY_INCLUDES = 

CCX = g++
CXXFLAGS = -g -O2  $(INC)

#============================================================
all: $(TARGET1)	$(TARGET2) $(TARGET3)

.o:.cpp	$(MY_INCLUDES)
	$(CCX)  -c  $(CXXFLAGS) $<  

$(TARGET1) :   $(C_OBJS1)
	$(CCX) $(CXXFLAGS)  $^ $(LIBDIRS)  -o $@

$(TARGET2) :   $(C_OBJS2)
	$(CCX) $(CXXFLAGS)  $^ $(LIBDIRS)  -o $@

$(TARGET3) :   $(C_OBJS3)
	$(CCX) $(CXXFLAGS)  $^ $(LIBDIRS)  -o $@

# Implicit rules: $@ = target name, $< = first prerequisite name, $^ = name of all prerequisites 
#============================================================

ALL_SOURCES = Makefile $(C_SOURCES) $(MY_INCLUDES)

NOTES =

clean:
	rm -f $(TARGET1) $(C_OBJS1) $(TARGET2) $(C_OBJS2) $(TARGET3) $(C_OBJS3) core  *~


