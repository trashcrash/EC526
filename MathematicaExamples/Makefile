.SUFFIXES:
.SUFFIXES: .o .cpp
#============================================================

TARGET1 = integrate
C_OBJS1 = integrate_sin.o


C_SOURCES =  integrate_sin.cpp 
MY_INCLUDES = quadmath.h

ALL_SOURCES = Makefile $(C_SOURCES) $(MY_INCLUDES)  scaling.dat std_out.dat NumericalIntegration_graph.nb

CCX = g++
CXXFLAGS = -g  -Wall
# -g -Wall 
#-std=gnu++11

#============================================================
all: $(TARGET1) 

.o:.cpp	$(MY_INCLUDES)
	$(CCX)  -c  $(CXXFLAGS) $<  

$(TARGET1) :   $(C_OBJS1)
	$(CCX) $(CXXFLAGS)  $^ $(LIBDIRS)  -o $@



# Implicit rules: $@ = target name, $< = first prerequisite name, $^ = name of all prerequisites
#============================================================

clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3)  $(C_OBJS1) $(C_OBJS2) $(C_OBJSTEST1) $(C_OBJ3) *~

tar: $(ALL_SOURCES) 
	tar cvf HW3_code.tar $(ALL_SOURCES)



