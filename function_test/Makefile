EXE:=main

FCCUDA:=nvfortran
FCMPI:=mpif90

FCFLAGS:=-O0 -traceback
SRC:= ./src
#FILES:= cudamod.o mpimod.o main.o
FILES:= cudamod.o gpu_fuc.o main.o 
MODS:=$(wildcard *.mod)

UNAME_S:=$(shell uname -n)
RM:=rm -fv

.SUFFIXES: .o .f .f90 .cuf

all: ${EXE}

${EXE}: ${FILES} ${MODS}
	${FCMPI} -cuda -o $@ ${FILES}

cudamod.o:
	${FCCUDA} ${FCFLAGS} -cuda -c ${SRC}/cudamod.cuf

gpu_fuc.o:
	${FCMPI} ${FCFLAGS} -cuda -c ${SRC}/gpu_fuc.f90 -lmpi
#mpimod.o:
#	${FCMPI} ${FCFLAGS} -cuda -c mpimod.f90 -lmpi

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