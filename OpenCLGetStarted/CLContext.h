#include <CL/cl.hpp>

typedef std::vector<cl::Device> DEVICELIST;

class CLContext
{
public:
	CLContext();
	cl::Context CLContext::setupContext();
	std::vector<cl::Device> setupDevices(cl::Context context);
	cl::Kernel setupKernel(cl::Program, std::string clFunction);
	cl::Program setupProgram(cl::Context context, std::vector<cl::Device> devices, std::string clFile);

	cl::Context getContext();
	DEVICELIST getDeviceList();
	cl::CommandQueue getQueue();
	cl::Kernel getKernel(int i);

private:
	cl::Context context;
	cl::Kernel kernel[3];
	cl::CommandQueue queue;
	DEVICELIST deviceList;
};