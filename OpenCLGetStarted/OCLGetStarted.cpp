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

#define THREADS_PER_WORKGROUP 32 /*must be a multiple of 2*/
int main(int argc, char **argv)
{

	int n = 1 << 20;
	int* t = getRandoms(n);


	/* First timer */
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	start = std::chrono::high_resolution_clock::now();

	/* CPU Sum */
	int sumArr = SumArray(t, n);

	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = (end - start);
	
	std::cout << "CPU SUM: " << sumArr << "\n";
	std::cout << "Timed: " << elapsed_seconds.count() << "s\n";


	int nbOfComputeUnits = 16;
	int chunkSize = n / nbOfComputeUnits;

	if (n%nbOfComputeUnits)
		chunkSize++;

	/* Platforms and devices */
	cl_int err;
	std::vector<cl::Platform> platformList;

	cl::Platform::get(&platformList);
	checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
	std::cerr << "Platform number is: " << platformList.size() << std::endl;

	std::string platformVendor;
	platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
	std::cerr << "Platform is by: " << platformVendor << "\n";

	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0 };
	cl::Context context(CL_DEVICE_TYPE_CPU, cprops, NULL, NULL, &err);
	checkErr(err, "Context::Context()");

	std::vector<cl::Device> devices;
	devices = context.getInfo<CL_CONTEXT_DEVICES>();
	checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

	/* Read file */
	std::ifstream file("Lesson1_kernels.cl");
	checkErr(file.is_open() ? CL_SUCCESS : -1, "lesson1_kernel.cl");
	std::string prog(std::istreambuf_iterator<char>(file),(std::istreambuf_iterator<char>()));

	/* Build program */
	cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length() + 1));
	cl::Program program(context, source);
	err = program.build(devices, "");
	checkErr(err, "Program::build()");
	cl::Kernel kernel(program, "SOLUTION", &err);

	/* Start second timer */
	std::chrono::time_point<std::chrono::high_resolution_clock> start2, end2;
	start2 = std::chrono::high_resolution_clock::now();

	/* buffer for array and sum */
	cl::Buffer buffer(context, CL_MEM_READ_WRITE, n*sizeof(int), NULL, &err);
	cl::Buffer sum(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
	

	/* command queue */
	cl::CommandQueue queue(context, devices[0], 0, &err);
	checkErr(err, "CommandQueue::CommandQueue()");

	int mysum = 0;
	queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, n*sizeof(int), t);
	queue.enqueueWriteBuffer(sum, CL_TRUE, 0, sizeof(int), &mysum);

	kernel.setArg(0, buffer);
	kernel.setArg(1, sum);
	kernel.setArg(2, chunkSize);
	kernel.setArg(3, n);

	cl::Event event;

	cl::NDRange local(THREADS_PER_WORKGROUP);
	cl::NDRange global(THREADS_PER_WORKGROUP*nbOfComputeUnits);


	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
	checkErr(err, "CommandQueue::enqueueNDRangeKernel()");

	end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<long double> elapsed_seconds2 = (end2 - start2);

	int* result = new int();
	err = queue.enqueueReadBuffer(sum, CL_TRUE, 0, sizeof(int), result);
	checkErr(err, "CommandQueue::enqueueReadBuffer()");
	
	std::cout << *result << "\n";
	std::cout << "Timed: " << elapsed_seconds2.count() << "s\n";
	std::cout << "Press any key to exit...\n";
	std::cin.ignore();
	return EXIT_SUCCESS;
}