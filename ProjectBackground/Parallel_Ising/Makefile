.SUFFIXES:
.SUFFIXES: .o .cpp

ACCEL_TYPE = PGI-multicore
#ACCEL_TYPE = PGI-tesla
#ACCEL_TYPE = PGI

CXX = pgc++
ifeq ($(ACCEL_TYPE),GCC)
	CXXFLAGS += -std=c99 -march=native -O3 -lm
else ifeq ($(ACCEL_TYPE),PGI)
        CXXFLAGS += -fast
else ifeq ($(ACCEL_TYPE),PGI-tesla)
        CXXFLAGS += -Minfo=accel -fast -O3 -acc -ta=tesla:fastmath -ta:tesla:fma -ta=tesla:cc60 
else ifeq ($(ACCEL_TYPE),PGI-multicore)
	CXXFLAGS += -Minfo=accel -fast -acc -ta=multicore
endif

#============================================================
TARGET	=  phi

C_SOURCES = phi.cpp ran2s.cpp rangs.cpp 
C_OBJS     =  phi.o ran2s.o rangs.o 
MY_INCLUDES =  ran2s.h

#============================================================
all: $(TARGET)

.o:.cpp	$(MY_INCLUDES) 
	$(CXX)  -c  $(CXXFLAGS) $<  

$(TARGET) :   $(C_OBJS) 
	$(CXX) $(CXXFLAGS)  $^ $(LIBDIRS)  -o $@

# Implicit rules: $@ = target name, $< = first prerequisite name, $^ = name of all prerequisites 
#============================================================

ALL_SOURCES = Makefile $(C_SOURCES) $(MY_INCLUDES)

NOTES =

clean:
	rm -f $(TARGET) $(C_OBJS) core*

TAGS:	$(ALL_SOURCES)
	etags $(ALL_SOURCES)

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGET).tar $(ALL_SOURCES)  $(NOTES)

$(TARGET).ps: $(ALL SOURCES)
	enscript -pcode.ps $(ALL_SOURCES)


