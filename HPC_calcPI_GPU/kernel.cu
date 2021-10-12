#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;


#define N 1024
#define blocks 128
#define threads 128

// подсчет числа пи через графический процессор
__global__ void gpu_calculation_pi(float* estimate, curandState* states) {
	// расчет глобального индекса потока
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points = 0;
	// координаты по осям
	float x_coordinates, y_coordinates;
	// The curand_init() function sets up an initial state allocated by the caller using the given seed,
	// sequence number, and offset within the sequence. Different seeds are guaranteed to produce different
	// starting states and different sequences. The same seed always produces the same state and the samesequence.
	// The state set up will be the state after 267 ⋅ sequence + offset calls to curand() from the seed state.
	curand_init(1234, tid, 0, &states[tid]);
	// расчет числа пи по алгоритму
	for (int i = 0; i < N; i++) {
		// This function returns a sequence of pseudorandom floats uniformly distributed between 0.0 and 1.0.
		// It may return from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded.
		// Distribution functions may use any number of unsigned integer values from a basic generator.
		// The number of values consumed is not guaranteed to be fixed.
		x_coordinates = curand_uniform(&states[tid]);
		y_coordinates = curand_uniform(&states[tid]);
		points += (pow(x_coordinates, 2) + pow(y_coordinates, 2) <= 1.0f);
	}
	estimate[tid] = 4.0f * points / (float)N;
}


// подсчет числа пи на центральном процессоре
float cpu_calculation_pi(long trials) {
	// кординаты декартовой системы координат
	float x_coordinates, y_coordinates;
	long points = 0.0f;
	// расчет числа пи по алгоритму
	for (long i = 0; i < trials; i++) {
		x_coordinates = rand() / (float)RAND_MAX;
		y_coordinates = rand() / (float)RAND_MAX;
		points += (pow(x_coordinates, 2) + pow(y_coordinates, 2) <= 1.0f);
	}
	return 4.0f * points / trials;
}

int main()
{
	// массивы для работы на GPU
	float host[blocks * threads];
	float* device;
	curandState* deviceStates;
	// переменная для подсчета времени работы алгоритма на GPU
	float gpu_time;
	// переменные для обработки событий времени на GPU
	cudaEvent_t gpu_start, gpu_stop;

	// запуск алгоритма на GPU
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	// выделение памяти для коипрования массивов на GPU
	cudaMalloc((void**)&device, blocks * threads * sizeof(float));
	cudaMalloc((void**)&deviceStates, threads * blocks * sizeof(curandState));
	// замеры времени работы алгоритма на GPU
	cudaEventRecord(gpu_start, 0);
	gpu_calculation_pi << <blocks, threads >> > (device, deviceStates);
	cudaEventRecord(gpu_stop, 0);
	// синхронизация
	cudaEventSynchronize(gpu_stop);
	// расчет затраченного времени
	cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
	// копирование полученных данных
	cudaMemcpy(host, device, blocks * threads * sizeof(float), cudaMemcpyDeviceToHost);
	float gpu_pi = 0.0f;
	for (int i = 0; i < blocks * threads; i++) {
		gpu_pi += host[i];
	}
	gpu_pi /= (blocks * threads);
	cout << "Approximate pi calculated on GPU is: " << gpu_pi << " and calculation took " << gpu_time << " msec" << endl;

	// высвобождени паамяти GPU
	cudaFree(device);
	cudaFree(deviceStates);

	// работа алгоритма на CPU
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	float cpu_pi = cpu_calculation_pi(blocks * threads * N);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double, std::milli> time_span = t2 - t1;							// расчет затраченного времени						
	double cpu_time = time_span.count();
	cout << "Approximate pi calculated on CPU is: " << cpu_pi << " and calculation took " << cpu_time << " msec" << endl;
	// расчет ускорения
	cout << "Acceleration of computations " << cpu_time/gpu_time;
	return 0;
}

