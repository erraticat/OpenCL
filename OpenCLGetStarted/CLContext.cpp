#include "CLContext.h"

#include <stdio.h>
#include <stdlib.h>
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

cl::Context CLContext::getContext()
{
	return this->context;
}

DEVICELIST CLContext::getDeviceList()
{
	return this->deviceList;
}

cl::CommandQueue CLContext::getQueue()
{
	return this->queue;
}

cl::Kernel CLContext::getKernel(int i)
{
	return this->kernel[i];
}

cl::Context CLContext::setupContext()
{
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
	return context;
}

std::vector<cl::Device> CLContext::setupDevices(cl::Context context)
{
	std::vector<cl::Device> devices;
	devices = context.getInfo<CL_CONTEXT_DEVICES>();
	checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");
	return devices;
}

cl::Program CLContext::setupProgram(cl::Context context, std::vector<cl::Device> devices, std::string clFile)
{
	/* Read file */
	cl_int err;
	std::ifstream file(clFile);
	checkErr(file.is_open() ? CL_SUCCESS : -1, clFile.c_str());
	std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
	file.close();

	/* Build program */
	cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length() + 1));
	cl::Program program(context, source);
	err = program.build(devices, "");
	std::string error("Program::build() " + clFile);
	checkErr(err, error.c_str());
	return program;
}

cl::Kernel CLContext::setupKernel(cl::Program program, std::string clFunction)
{
	cl_int err;
	cl::Kernel kernel(program, clFunction.c_str(), &err);
	std::string error("CLContext::setupKernel " + clFunction);
	checkErr(err, error.c_str());
	return kernel;
}

CLContext::CLContext()
{
	cl_int err;
	this->context = setupContext();
	this->deviceList = setupDevices(this->context);
	cl::Program prog = setupProgram(this->context, this->deviceList, "coursework_kernel.cl");

	this->kernel[0] = setupKernel(prog, "ComputeCSumStage1");
	this->kernel[1] = setupKernel(prog, "ComputeCSumStage2");
	this->kernel[2] = setupKernel(prog, "ComputeCSumStage3");


	/* command queue */
	this->queue = cl::CommandQueue(context, this->deviceList[0], 0, &err);
	checkErr(err, "CommandQueue::CommandQueue()");
}