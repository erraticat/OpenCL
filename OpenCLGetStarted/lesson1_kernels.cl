#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable


#define THREADS_PER_WORKGROUP 32 /*must be a multiple of 2*/

__kernel void DEFAULT_KERNEL  (__global int *array, __global int *result)
{																						
	atomic_add (result,array[get_global_id(0)]);
}													


__kernel void SOLUTION  (__global int *array, __global int *result, int chunkSize, int nbIntegers)

{
	// step1, each compute unit computes THREADS_PER_WORKGROUP sums  (1 result per thread)

	__local int sums[THREADS_PER_WORKGROUP];//Creates shared memory for all threads within the group. Shared Mem is fast. THREADS_PER_WORKGROUP work items per compute unit (must be the same as host code)
	sums[get_local_id(0)]=0;
	for (int i=0;i<chunkSize;i+=THREADS_PER_WORKGROUP)
	{
		int index=get_local_id(0)+i+get_group_id(0)*chunkSize;
		if ((get_local_id(0)+i<chunkSize)&&(index<nbIntegers)) //tests whether the index is still with the chunk of data to be processed by the compute unit. test is needed if our problem is not a multiple of THREADS_PER_WORKGROUP.
			sums[get_local_id(0)]+=array[index];
	}
	// step2, 256 sums are merged into 1 locally (1 result per work group). It is important to note that it is better to do it this way on many platforms as access to local variables is very fast (when compared to global mem, or kernel calls).
	barrier(CLK_LOCAL_MEM_FENCE);//This point should be reached by all threads, deadlock otherwise! Needed because step1 must be completed before step 2
	int groupsize=THREADS_PER_WORKGROUP/2;
	while(groupsize>0)
	{
		if ((get_local_id(0)<groupsize))  /*Some of the local threads will become inactive */ 
		{
			sums[get_local_id(0)]+=sums[get_local_id(0)+groupsize];	
		}
		groupsize>>=1; //divide the number of threads by two;
		barrier(CLK_LOCAL_MEM_FENCE);//All threads should be done before completing the next iteration.
	}																				
	// step 3: Each compute unit does one single atomic add (final result. Atomic are costly, but only done for the few workgroups started) 
	if (get_local_id(0)==0) 
		atomic_add ((result),sums[0]); //still atomic add, but done only nbIntegers/512 times
}