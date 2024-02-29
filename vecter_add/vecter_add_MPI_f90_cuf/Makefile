EXE:=main

FCCUDA:=nvfortran
FCMPI:=mpif90

FCFLAGS:=-O0 -traceback

FILES:= kernel.o mpimod.o main.o 

MODS:=$(wildcard *.mod)

UNAME_S:=$(shell uname -n)
RM:=rm -fv

.SUFFIXES: .o .f .f90 .cuf

all: ${EXE}

${EXE}: ${FILES} ${MODS}
	${FCMPI} -cuda -o $@ ${FILES}

kernel.o:
	${FCCUDA} ${FCFLAGS} -cuda -c kernel.cuf

mpimod.o:
	${FCMPI} ${FCFLAGS} -cuda -c mpimod.f90 -lmpi

main.o:
	${FCMPI} ${FCFLAGS} -cuda -c main.f90 -lmpi

%.mod: %.f90
	@echo “Some modules are out of date. Do clean and then recompile”
	${RM} $@ ${EXE}

.PHONY: clean

clean:
	${RM} *.o
	${RM} *.mod
	${RM} ${EXE}