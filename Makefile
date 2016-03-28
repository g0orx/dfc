LINK=nvcc
CC=nvcc

NVCFLAGS=-Xptxas -v

INCLUDES=-I/usr/local/cuda/targets/armv7-linux-gnueabihf/include -I/usr/local/cuda/samples/common/inc

LIBS=-L/usr/local/cuda/targets/armv7-linux-gnueabihf/lib -lcufft -lpthread -lrt -lpulse-simple -lpulse

PROGRAM=dfc

SOURCES=start.cu \
        time.cu \
	coeff.cu \
	filters.cu \
	inputbuffer.cu \
	rawbuffer.cu \
	pcie.cu \
	socket.cu \
	file.cu \
	dfc.cu \
	hermes.cu \
	new_protocol.cu \
	receiver.cu \
	audio.cu \
	main.cu

HEADERS=common.cuh \
	time.cuh \
        coeff.cuh \
        filters.cuh \
        inputbuffer.cuh \
        rawbuffer.cuh \
	pcie.cuh \
	socket.cuh \
	file.cuh \
	dfc.cuh \
        new_protocol.cuh \
        hermes.cuh \
        receiver.cuh \
        audio.cuh

OBJS=start.o \
        time.o \
	coeff.o \
	filters.o \
	inputbuffer.o \
	rawbuffer.o \
	pcie.o \
	socket.o \
	file.o \
	dfc.o \
	hermes.o \
	new_protocol.o \
	receiver.o \
	audio.o \
	main.o

all: $(PROGRAM) $(HEADERS) $(SOURCES)

$(PROGRAM): $(OBJS)
	$(LINK) -o $(PROGRAM) $(OBJS) $(LIBS)

%.o: %.cu
	$(CC) $(NVCFLAGS) $(CUDA_ARCH) $(INCLUDES) -c -o $@ $<

clean:
	$(RM) -rf *.o
	$(RM) -rf $(PROGRAM)
