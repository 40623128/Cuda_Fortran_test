EXE:=mpigpu

FCCUDA:=nvfortran
FCMPI:=mpif90

FCFLAGS:=-O0 -traceback

FILES:= cudamod.o main.o

MODS:=$(wildcard *.mod)

UNAME_S:=$(shell uname -n)
RM:=rm -fv

.SUFFIXES: .o .f .f90 .cuf

all: ${EXE}

${EXE}: ${FILES} ${MODS}
	${FCMPI} -cuda -o $@ ${FILES}

cudamod.o:
	${FCCUDA} ${FCFLAGS} -cuda -c cudamod.cuf


main.o:
	${FCMPI} ${FCFLAGS} -cuda -c main.f90

%.mod: %.f90
	@echo “Some modules are out of date. Do clean and then recompile”
	${RM} $@ ${EXE}

.PHONY: clean

clean:
	${RM} *.o
	${RM} *.mod
	${RM} ${EXE}