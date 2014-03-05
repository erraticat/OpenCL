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
#include "Chrono.h"
#include "CLContext.h"

#define DATA_SIZE 8
#define NUMBER_GROUPS 8

#define PRINT_ARRAYS false  // Set this to true to print out the contents of both arrays

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

void compareArrays(int* a, int* b, int n)
{
	for (int i = 0; i < n; i++)
	{
		if (a[i] != b[i])
		{
			std::cout << "ERROR Element " << i << " NO MATCH\n";
			return;
		}
	}
	std::cout << "Arrays Match \n";
}

void cpuCumulativeSum(int* a, int* b, int n)
{
	b[0] = a[0];
	for (int i = 1; i < n; i++)
	{
		//if (i % 8 == 0) { b[i] = a[i]; continue; } // test array without stage 3
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
	/*
	* N items processed by N/2 workgroups will require 2 iterations per workgroup.
	*/
	cl_int err = 0;
	int n = 512; // Number of Elements.  Must be a multiple of (numberGroups * threadsPerGroup) or 64 by default
	int numberThreads = DATA_SIZE * NUMBER_GROUPS; //64 by default


	int chunkSize = n / NUMBER_GROUPS; // number of elements to give each group
	int numChunks = n / DATA_SIZE; //total number of chunks that will be handled by all workgroups

	int* arrayA = generateLinearArray(n);
	int* arrayB = new int[n]();

	Chrono c;
	cpuCumulativeSum(arrayA, arrayB, n);
	c.PrintElapsedTime_us("ORIGINAL TIME: ");

	if (PRINT_ARRAYS)
	{
		std::cout << "CPU RESULT: ";
		printArray(arrayB, n);
		std::cout << "\n";
	}

	int* arrayB2 = new int[n]();

	int* arrayC;
	arrayC = new int[numChunks]();
	
	CLContext* ctx = new CLContext();
	cl::Context clctx = ctx->getContext();

	cl::Buffer bufferA(clctx, CL_MEM_READ_WRITE, n*sizeof(int), NULL, &err);
	//checkErr(err, "cl::Buffer 1");
	cl::Buffer bufferB(clctx, CL_MEM_READ_WRITE, n*sizeof(int), NULL, &err);
	//checkErr(err, "cl::Buffer 2");
	cl::Buffer bufferC(clctx, CL_MEM_READ_WRITE, numChunks*sizeof(int), NULL, &err);
	//checkErr(err, "cl::Buffer 3");

	cl::CommandQueue queue = ctx->getQueue();
	queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, n*sizeof(int), arrayA);
	queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, n*sizeof(int), arrayB2);
	queue.enqueueWriteBuffer(bufferC, CL_TRUE, 0, numChunks*sizeof(int), arrayC);

	for (int i = 0; i < 3; i++)
	{
		cl::Kernel kernel = ctx->getKernel(i);
		kernel.setArg(2, chunkSize);
		kernel.setArg(3, numberThreads);
	}

	ctx->getKernel(0).setArg(0, bufferA);
	ctx->getKernel(0).setArg(1, bufferB);

	ctx->getKernel(1).setArg(0, bufferB);
	ctx->getKernel(1).setArg(1, bufferC);

	ctx->getKernel(2).setArg(0, bufferC);
	ctx->getKernel(2).setArg(1, bufferB);

	int numberofgroups = 1;
	int threadspergroup = n;

	cl::NDRange global(numberThreads);
	cl::NDRange local(DATA_SIZE);

	Chrono c2;
	err = queue.enqueueNDRangeKernel(ctx->getKernel(0), cl::NullRange, global, local);
	//checkErr(err, "CommandQueue::enqueueNDRangeKernel() || Stage 1");

	err = queue.enqueueNDRangeKernel(ctx->getKernel(1), cl::NullRange, global, local);
	//checkErr(err, "CommandQueue::enqueueNDRangeKernel() || Stage 2");

	err = queue.enqueueNDRangeKernel(ctx->getKernel(2), cl::NullRange, global, local);
	//checkErr(err, "CommandQueue::enqueueNDRangeKernel() || Stage 3");

	c2.PrintElapsedTime_us("\nOPENCL TIME: ");

	err = queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, numChunks*sizeof(int), arrayC);
	checkErr(err, "CommandQueue::enqueueReadBuffer() C");

	if (PRINT_ARRAYS)
	{
		std::cout << "\nContents of C: \n";
		printArray(arrayC, numChunks);
	}

	err = queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, n*sizeof(int), arrayB2);
	checkErr(err, "CommandQueue::enqueueReadBuffer() B");

	if (PRINT_ARRAYS)
	{
		std::cout << "\nOPENCL RESULT: ";
		printArray(arrayB2, n);
		std::cout << "\n\n";
	}

	compareArrays(arrayB, arrayB2, n);
	getchar();
}