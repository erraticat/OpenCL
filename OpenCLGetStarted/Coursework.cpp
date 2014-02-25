#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.hpp>
#include <iterator>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstdio>

#include <utility>

#include <chrono>
#include <ctime>

#include "CLContext.h"

inline void checkErr(cl_int err, const char* name)
{
	if (err != CL_SUCCESS)
	{
		std::cerr << "Error: " << name << " (" << err << ")" << std::endl;
		std::cin.ignore();
		exit(EXIT_FAILURE);
	}
}

int* getRandoms(int n)
{
	int *t = new int[n];
	for (int i = 0; i<n; i++)
		t[i] = (rand()) % 100;
	return t;
}


int SumArray(int *t, int n)
{
	int sum = 0;
	for (int i = 0; i<n; i++)
		sum += t[i];
	return sum;
}

int* generateLinearArray(int n)
{
	int* result = new int[n]();
	for (int i = 0; i < n; i++)
		result[i] = i;

	return result;
}

void cpuCumulativeSum(int* a, int* b, int n)
{
	b[0] = a[0];
	for (int i = 1; i < n; i++)
	{
		b[i] = a[i] + b[i - 1];
	}
}

void printArray(int* array, int n)
{
	for (int i = 0; i < n; i++)
		std::cout << array[i] << " ";

	std::cout << "\n";
}

#define THREADS_PER_WORKGROUP 32 /*must be a multiple of 2*/

int main(int argc, char **argv)
{
	cl_int err = 0;

	int n = 8;

	int* arrayA = generateLinearArray(n);
	int* arrayB = new int[n]();

	cpuCumulativeSum(arrayA, arrayB, n);

	std::cout << "Result: ";
	printArray(arrayB, n);

	delete arrayB;
	arrayB = new int[n]();
	
	CLContext* ctx = new CLContext();
	cl::Context clctx = ctx->getContext();

	cl::Buffer bufferA(clctx, CL_MEM_READ_WRITE, n*sizeof(int), NULL, &err);
	checkErr(err, "cl::Buffer 1");
	cl::Buffer bufferB(clctx, CL_MEM_READ_WRITE, n*sizeof(int), NULL, &err);
	checkErr(err, "cl::Buffer 2");

	cl::CommandQueue queue = ctx->getQueue();
	queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, n*sizeof(int), arrayA);
	queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, n*sizeof(int), arrayB);

	int chunkSize = 8;
	int dataSize = 1;

	for (int i = 0; i < 3; i++)
	{
		cl::Kernel kernel = ctx->getKernel(i);
		kernel.setArg(0, bufferA);
		kernel.setArg(1, bufferB);
		kernel.setArg(2, chunkSize);
		kernel.setArg(3, dataSize);
	}

	int numberofgroups = 1;
	int threadspergroup = n;

	cl::NDRange global(8);
	cl::NDRange local(8);

	err = queue.enqueueNDRangeKernel(ctx->getKernel(0), cl::NullRange, global, local);
	checkErr(err, "CommandQueue::enqueueNDRangeKernel() || Stage 1");

	/*
	err = queue.enqueueNDRangeKernel(ctx->getKernel(1), cl::NullRange, global, local);
	checkErr(err, "CommandQueue::enqueueNDRangeKernel() || Stage 2");

	err = queue.enqueueNDRangeKernel(ctx->getKernel(2), cl::NullRange, global, local);
	checkErr(err, "CommandQueue::enqueueNDRangeKernel() || Stage 3");
	*/

	err = queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, n*sizeof(int), arrayB);
	checkErr(err, "CommandQueue::enqueueReadBuffer()");

	std::cout << "OPENCL RESULT: ";
	printArray(arrayB, n);

	getchar();
}