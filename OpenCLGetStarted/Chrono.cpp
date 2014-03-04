
// Chrono.cpp: implementation of the Chrono class.
//
//////////////////////////////////////////////////////////////////////
//#include "stdafx.h"
#include <windows.h> 
#include "Chrono.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
LARGE_INTEGER frequency;

Chrono::Chrono()
{
	start=0;
	QueryPerformanceFrequency(&frequency);
	InitChrono();
}

Chrono::~Chrono()
{

}

void Chrono::InitChrono()
{
#ifdef WIN32
		//start=timeGetTime();
		LARGE_INTEGER t;
		start=QueryPerformanceCounter(&t);
		start=t.QuadPart;
#else
		start=clock();
#endif
}

long long Chrono::Elapsed_ms()
{		
#ifdef WIN32
		LARGE_INTEGER t;

		/*finish=timeGetTime();
		return  (finish - start);*/
		QueryPerformanceCounter(&t);
		finish=t.QuadPart;
		return  ((((long long) 1000)*(finish - start)))/frequency.QuadPart;
#else
		finish=clock();
		return (long) ((((double) (finish - start))*1000)/CLOCKS_PER_SEC);
#endif
}

long long Chrono::Elapsed_us()
{		
#ifdef WIN32
		LARGE_INTEGER t;

		/*finish=timeGetTime();
		return  (finish - start);*/
		QueryPerformanceCounter(&t);
		finish=t.QuadPart;
		return  ((((long long) 1000000)*(finish - start)))/frequency.QuadPart;
#else
		finish=clock();
		return (long) ((((double) (finish - start))*1000)/CLOCKS_PER_SEC);
#endif
}

long long Chrono::Time_us()
{		
#ifdef WIN32
		LARGE_INTEGER t;
		QueryPerformanceCounter(&t);
		long long l=t.QuadPart;
		return  (1000000*l)/frequency.QuadPart;
#else
		finish=clock();
		return (long) ((((double) (finish - start))*1000)/CLOCKS_PER_SEC);
#endif
}

long long Chrono::Time_ms()
{		
#ifdef WIN32
		LARGE_INTEGER t;
		QueryPerformanceCounter(&t);
		long long l=t.QuadPart;
		return  (1000*l)/frequency.QuadPart;
#else
		finish=clock();
		return (long) ((((double) (finish - start))*1000)/CLOCKS_PER_SEC);
#endif
}


void Chrono::PrintProgressTime(char *s, int nb_process)
{
	long long t=Elapsed_ms()/(nb_process);
	printf("%s%d.%d%d%d s\n", s, t/1000, (t%1000)/100,(t%100)/10,t%10);
}	

void Chrono::PrintElapsedTime(char *s, int nb_process)
{
	long long t=Elapsed_ms()/(nb_process);
	printf("%s%d.%d%d%d s\n", s, t/1000, (t%1000)/100,(t%100)/10,t%10);
	InitChrono();
}

void Chrono::PrintElapsedTime_us(char *s)
{
	long long t=Elapsed_us();
	printf("%s%d\n", s,t);
	InitChrono();
}

char *Chrono::ReturnTimeString()
{
	long long i,a;
	long long t=Elapsed_ms();
	a=t;
	i=-1;
	while(a>0) 
	{
		i++;
		a/=10;
	}
	a=i;
	while (i>=0)
	{
		Sentence[i]=((char) (t%10))+'0';
		t/=10;
		i--;
	}
	Sentence[a+1]=' ';	
	Sentence[a+2]='m';
	Sentence[a+3]='s';
	Sentence[a+4]=0;
	return Sentence;
}
