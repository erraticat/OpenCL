#define THREADS_PER_GROUP 8

__kernel void ComputeCSumStage1  (__global int* A, __global int* B, __global int* C, int chunkSize, int n)
{
	int loc = get_local_id(0);
	int glob = get_global_id(0);

	int bloc = loc + 1;
	if (bloc % 2 == 0)
	{
		B[glob] = A[glob] + A[glob-1];
		B[glob-1] = A[glob-1];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (bloc % 4 == 0 || bloc % 6 == 0 && loc !=0)
	{
		B[glob] += B[glob-2];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (bloc % 8 == 0 && loc != 0)
	{
		B[glob] += B[glob-4];
	}
	if (loc % 2 == 0 && loc != 0)
	{
		B[loc] += B[loc-1];
	}
	

}																					

__kernel void ComputeCSumStage2  (__global int* A, __global int* B, int chunkSize, int n)
{
																						
}													

__kernel void ComputeCSumStage3  (__global int* A, __global int* B,  int chunkSize, int n)
{
																						
}	