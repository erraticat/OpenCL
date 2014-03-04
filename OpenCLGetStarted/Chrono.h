#pragma once
//////////////////////////////////////////////////////////////////////
// Chrono.h: interface for the Chrono class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(CHRONO_H)
#define CHRONO_H

#ifdef WIN32
#include <windows.h>
#include <mmsystem.h>
#else
#include <time.h>
#endif
#include <stdio.h>

#if _MSC_VER > 1000 
#pragma once
#endif // _MSC_VER > 1000

class Chrono  
{
private:
#ifdef WIN32
	volatile long long start, finish;
#else
	clock_t start, finish;
#endif
	char Sentence[128];
protected:
	static long long Time_ms();
public:
	Chrono();
	virtual ~Chrono();
	void InitChrono();
	long long Time_us();
	long long Elapsed_ms();
	long long Elapsed_us();
	void PrintElapsedTime(char *s, int nb_process=1);
	void PrintElapsedTime_us(char *s);
	void PrintProgressTime(char *s, int nb_process=1);
	char *ReturnTimeString();

};

#endif 


