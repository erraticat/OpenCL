#define THREADS_PER_GROUP 8

void prefixSumBlockEight(__global int* A, __global int* B, int offset)
{
	__local int shArray[THREADS_PER_GROUP];
	int loc = get_local_id(0);
	int glob = get_local_id(0)+offset;
	shArray[loc]=0;

	if (loc == 0)
	{
		shArray[loc] = A[glob];
	}
	else
	{
		shArray[loc] = A[glob] + A[glob-1];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	__local int shArray2[THREADS_PER_GROUP];
	if (loc > 1)
	{
		shArray2[loc] = shArray[loc]+shArray[loc-2];
	}
	else
	{
		shArray2[loc] = shArray[loc];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (loc > 3)
	{
		B[glob] = shArray2[loc] + shArray2[loc-4];
	}
	else
	{
		B[glob] = shArray2[loc];
	}
}

__kernel void ComputeCSumStage1  (__global int* A, __global int* B, int chunkSize, int n)
{
	int loc = get_local_id(0);
	int glob = get_global_id(0);

	int iterations = chunkSize / THREADS_PER_GROUP;

	int groupOffset = iterations * THREADS_PER_GROUP * get_group_id(0);
	for (int i = 0; i < iterations; i++)
	{
		prefixSumBlockEight(A, B, (i*THREADS_PER_GROUP) + groupOffset);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}																					

__kernel void ComputeCSumStage2  (__global int* A, __global int* B, int chunkSize, int n)
{
		int iterations = chunkSize / THREADS_PER_GROUP;
		int groupOffset = iterations * THREADS_PER_GROUP * get_group_id(0);

		if (get_local_id(0)+1 % THREADS_PER_GROUP != 0)
		{
			return;
		}

		for (int i = 0; i < iterations; i++)
		{		
			B[get_group_id(0)*iterations+i]=A[(i+1)*THREADS_PER_GROUP-1+groupOffset];
		}			
}													

__kernel void ComputeCSumStage3  (__global int* A, __global int* B,  int chunkSize, int n)
{
		int loc = get_local_id(0);
		int glob = get_global_id(0);
		int grp = get_group_id(0);
		
		int iterations = chunkSize / THREADS_PER_GROUP;
		int groupOffset = iterations * THREADS_PER_GROUP * get_group_id(0);

		for (int i = 0; i < iterations; i++)
		{
			int innergrp = grp * iterations+i;
			for (int x = innergrp; x > 0; x--)
			{
				B[loc+groupOffset+(i*8)]+=A[(x-1)];
			}
		}												
}	