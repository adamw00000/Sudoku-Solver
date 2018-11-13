
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <set>


#define SIZE 9

typedef uint8_t byte;

struct Sudoku
{
	byte board[SIZE][SIZE];
	uint32_t constraintStructures[SIZE];

	__host__ __device__ Sudoku() { }

	__host__ __device__ Sudoku(byte board[SIZE][SIZE], uint32_t constraintStructures[SIZE])
	{
		for (int i = 0; i < SIZE; i++)
		{
			this->constraintStructures[i] = constraintStructures[i];
			for (int j = 0; j < SIZE; j++)
				this->board[i][j] = board[i][j];
		}
	}
};

__host__ __device__ void PrintSudoku(byte arr[SIZE][SIZE])
{
	char s[200];
	int k = 0;
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			s[k] = arr[i][j] + '0';
			s[k + 1] = j == SIZE - 1 ? '\n' : ' ';
			k += 2;
		}
	}
	s[k] = '\n';
	s[k + 1] = '\0';
	printf("%s", s);
}

__host__ __device__ void BoardToString(byte arr[SIZE][SIZE], char* s)
{
	int k = 0;
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			s[k] = arr[i][j] + '0';
			s[k + 1] = j == SIZE - 1 ? '\n' : ' ';
			k += 2;
		}
	}
	s[k] = '\n';
	s[k + 1] = '\0';
}

__host__ __device__ bool IsNumberInConstraintStructure(byte number, const uint32_t& rowStructure, const uint32_t& colStructure, const uint32_t& cellStructure)
{
	return (rowStructure & (1U << number) || colStructure & (1U << (SIZE + number)) || cellStructure & (1U << (SIZE + SIZE + number)));
}

__host__ __device__ void AddNumberToConstraintStructure(byte number, uint32_t& rowStructure, uint32_t& colStructure, uint32_t& cellStructure)
{
	if (number != 0)
	{
		rowStructure |= (1U << number);
		colStructure |= (1U << (SIZE + number));
		cellStructure |= (1U << (SIZE + SIZE + number));
	}
}

__host__ __device__ void RemoveNumberFromRowOrColumn(byte number, uint32_t& rowStructure, uint32_t& colStructure, uint32_t& cellStructure)
{
	if (number != 0)
	{
		rowStructure &= ~(1U << number);
		colStructure &= ~(1U << (SIZE + number));
		cellStructure &= ~(1U << (SIZE + SIZE + number));
	}
}

__host__ __device__ byte cell(byte i, byte j)
{
	return (i / 3) * 3 + j / 3;
}

__global__ void activeResetKernel(int* d_active, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i >= n)
		return;

	d_active[i] = false;
}

__global__ void copyKernel(Sudoku* d_sudokus, Sudoku* d_sudokus_target, int* d_active, int* d_active_scan, int* d_active_copy, int n, int newMax)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i == 0)
		return;

	if (i < n && d_active_copy[i] == true)
	{
		memcpy(d_sudokus_target + d_active_scan[i], d_sudokus + i, sizeof(Sudoku));
		d_active[d_active_scan[i]] = true;
	}
}

__global__ void copyActiveKernel(int* d_active, int* d_active_copy, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < n)
	{
		d_active_copy[i] = d_active[i];
	}
}

__global__ void sudokuKernel(Sudoku* d_sudokus, int* d_active, int n, byte* currentField)
{
	int i = (blockIdx.x * blockDim.x) +  threadIdx.x;

	if (i > n)
		return;

	if (d_active[0] == true)
	{
		return;
	}

	Sudoku mySudoku;
	memcpy(&mySudoku, d_sudokus + i, sizeof(Sudoku));

	if (d_active[i] == false)
	{
		return;
	}

	if ((*currentField) == (byte)-1)
	{
		memcpy(d_sudokus, &mySudoku, sizeof(Sudoku));
		d_active[0] = true;
		return;
	}

	byte row = (*currentField) / SIZE;
	byte col = (*currentField) % SIZE;
	byte cellnr = cell(row, col);

	for (byte number = 1; number <= SIZE; number++)
	{
		if (!IsNumberInConstraintStructure(number, mySudoku.constraintStructures[row], mySudoku.constraintStructures[col], mySudoku.constraintStructures[cellnr]))
		{
			mySudoku.board[row][col] = number;
			AddNumberToConstraintStructure(number, mySudoku.constraintStructures[row], mySudoku.constraintStructures[col], mySudoku.constraintStructures[cellnr]);

			int index = n + 1 + (i - 1) * SIZE + (number - 1);
			memcpy(d_sudokus + index, &mySudoku, sizeof(Sudoku));
			d_active[index] = true;

			RemoveNumberFromRowOrColumn(number, mySudoku.constraintStructures[row], mySudoku.constraintStructures[col], mySudoku.constraintStructures[cellnr]);
			mySudoku.board[row][col] = 0;
		}
	}
	d_active[i] = false;
}

int ReadSudoku(byte board[SIZE][SIZE], std::string filename)
{
	std::ifstream stream;
	stream.open(filename.c_str());

	for (int i = 0; i < SIZE; i++) 
	{
		for (int j = 0; j < SIZE; j++)
		{
			char c = stream.get();
			while (stream.good() && c == '\n')
				c = stream.get();

			if (!stream.good())
			{
				printf("%s - Invalid file format!\n", filename.c_str());
				return -1;
			}
			if (c != 'x' && c != '0')
			{
				int n = atoi(&c);
				board[i][j] = n;
			}
			else
				board[i][j] = 0;
		}
	}

	stream.close();
	return 0;
}

void GetRowColAndCellNumbers(byte sudoku[SIZE][SIZE], uint32_t constraintStructures[])
{
	for (byte i = 0; i < SIZE; i++)
	{
		constraintStructures[i] = 0;
	}

	for (byte i = 0; i < SIZE; i++)
	{
		for (byte j = 0; j < SIZE; j++)
		{
			if (sudoku[i][j] != 0)
			{
				AddNumberToConstraintStructure(sudoku[i][j], constraintStructures[i], constraintStructures[j], constraintStructures[cell(i, j)]);
			}
		}
	}
}

void GetEmptyFields(byte sudoku[SIZE][SIZE], byte emptyFields[SIZE * SIZE])
{
	int k = 0;
	for (byte i = 0; i < SIZE; i++)
	{
		for (byte j = 0; j < SIZE; j++)
		{
			if (sudoku[i][j] == 0)
			{
				emptyFields[k] = i * SIZE + j;
				k++;
			}
		}
	}
	for (; k < SIZE * SIZE; k++)
	{
		emptyFields[k] = -1;
	}
}

cudaError_t SolveSudoku(byte sudokuArray[SIZE][SIZE])
{
	uint32_t constraintStructures[SIZE];
	byte emptyFields[SIZE * SIZE];

	GetRowColAndCellNumbers(sudokuArray, constraintStructures);
	GetEmptyFields(sudokuArray, emptyFields);

	cudaError_t cudaStatus;

	Sudoku activeSudoku(sudokuArray, constraintStructures);

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int activeBlocks = 1;
	Sudoku *d_sudokus;
	Sudoku *d_sudokus_target;

	Sudoku *h_sudokus = (Sudoku*)malloc((activeBlocks + 1) * sizeof(Sudoku));
	if (h_sudokus == NULL) {
		fprintf(stderr, "malloc failed!");
		return cudaStatus;
	}
	
	byte* d_active_field;
	
	int *d_active;
	int *d_active_copy;
	int *d_active_scan;

	int *h_active = (int*)malloc((activeBlocks + 9 * activeBlocks + 1) * sizeof(int));
	if (h_active == NULL) {
		fprintf(stderr, "malloc failed!");
		return cudaStatus;
	}

	for (int i = 0; i < (activeBlocks + 1); i++) {
		if (i == 1)
		{
			memcpy(h_sudokus + i, &activeSudoku, sizeof(Sudoku));
		}
	}

	for (int i = 0; i < (activeBlocks + 9 * activeBlocks + 1); i++) {
		if (i == 1)
		{
			h_active[i] = true;
		}
		else
		{
			h_active[i] = false;
		}
	}

	cudaStatus = cudaMalloc((void**)&d_sudokus, (activeBlocks + 9 * activeBlocks + 1) * sizeof(Sudoku));
	if (cudaStatus != cudaSuccess) {
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("Time for the kernel: %f ms\n", time);

		printf("Not enough memory to solve this sudoku!\n");
		//fprintf(stderr, "cudaMalloc failed!");
		free(h_active);
		free(h_sudokus);
		cudaFree(d_sudokus);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(d_sudokus, h_sudokus, (activeBlocks + 1) * sizeof(Sudoku), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		free(h_active);
		free(h_sudokus);
		cudaFree(d_sudokus);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&d_active, (activeBlocks + 9 * activeBlocks + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("Time for the kernel: %f ms\n", time);
		printf("Not enough memory to solve this sudoku!\n");
		//fprintf(stderr, "cudaMalloc failed!");
		free(h_active);
		free(h_sudokus);
		cudaFree(d_sudokus);
		cudaFree(d_active);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&d_active_scan, (activeBlocks + 9 * activeBlocks + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("Time for the kernel: %f ms\n", time);
		printf("Not enough memory to solve this sudoku!\n");
		//fprintf(stderr, "cudaMalloc failed!");
		free(h_active);
		free(h_sudokus);
		cudaFree(d_sudokus);
		cudaFree(d_active);
		cudaFree(d_active_scan);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(d_active, h_active, (activeBlocks + 9 * activeBlocks + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		free(h_active);
		free(h_sudokus);
		cudaFree(d_sudokus);
		cudaFree(d_active);
		cudaFree(d_active_scan);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&d_active_field, sizeof(byte));
	if (cudaStatus != cudaSuccess) {
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("Time for the kernel: %f ms\n", time);

		printf("Not enough memory to solve this sudoku!\n");
		//fprintf(stderr, "cudaMalloc failed!");
		free(h_active);
		free(h_sudokus);
		cudaFree(d_sudokus);
		cudaFree(d_active);
		cudaFree(d_active_scan);
		cudaFree(d_active_field);
		return cudaStatus;
	}

	thrust::device_ptr<int> dev_active_ptr(d_active);
	thrust::device_ptr<int> dev_active_scan_ptr(d_active_scan);

	int i = 0;
	bool lastActive;
	while (1)
	{
		if (activeBlocks <= 0)
		{
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			printf("Time for the kernel: %f ms\n", time);

			printf("Sudoku:\n");
			PrintSudoku(sudokuArray);
			printf("This sudoku doesn't have a solution!\n");
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(d_active_field, emptyFields + i, sizeof(byte), cudaMemcpyHostToDevice);
		i++;
		printf("Iteration: %d, active blocks - %d\n", i, activeBlocks);

		sudokuKernel <<<(activeBlocks + 1)/1024 + 1, 1024>>>(d_sudokus, d_active, activeBlocks, d_active_field);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "sudokuKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				free(h_active);
				free(h_sudokus);
				cudaFree(d_sudokus);
				cudaFree(d_active);
				cudaFree(d_active_scan);
				cudaFree(d_active_field);
				return cudaStatus;
			}

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sudokuKernel!\n", cudaStatus);
				free(h_active);
				free(h_sudokus);
				cudaFree(d_sudokus);
				cudaFree(d_active);
				cudaFree(d_active_scan);
				cudaFree(d_active_field);
				return cudaStatus;
			}

		if (dev_active_ptr[0] == true)
		{
			printf("Solution found!\n");
			break;
		}

		activeBlocks = activeBlocks + activeBlocks * 9;

		dev_active_ptr[0] = true;
		thrust::exclusive_scan(dev_active_ptr, dev_active_ptr + activeBlocks + 1, dev_active_scan_ptr);
		dev_active_ptr[0] = false;
		int newActive = thrust::max_element(dev_active_scan_ptr, dev_active_scan_ptr + activeBlocks + 1)[0];

		lastActive = dev_active_ptr[activeBlocks];
		if (lastActive)
			newActive++;

		cudaStatus = cudaMalloc((void**)&d_sudokus_target, (activeBlocks + 9 * activeBlocks + 1) * sizeof(Sudoku));
		if (cudaStatus != cudaSuccess) {
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			printf("Time for the kernel: %f ms\n", time);
			printf("Not enough memory to solve this sudoku!\n");
			//fprintf(stderr, "cudaMalloc failed!");
			free(h_active);
			free(h_sudokus);
			cudaFree(d_sudokus);
			cudaFree(d_sudokus_target);
			cudaFree(d_active);
			cudaFree(d_active_scan);
			cudaFree(d_active_field);
			return cudaStatus;
		}

		cudaStatus = cudaMalloc((void**)&d_active_copy, (activeBlocks + 1) * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			printf("Time for the kernel: %f ms\n", time);
			printf("Not enough memory to solve this sudoku!\n");
			//fprintf(stderr, "cudaMalloc failed!");
			free(h_active);
			free(h_sudokus);
			cudaFree(d_sudokus);
			cudaFree(d_sudokus_target);
			cudaFree(d_active);
			cudaFree(d_active_scan);
			cudaFree(d_active_copy);
			cudaFree(d_active_field);
			return cudaStatus;
		}
		
		copyActiveKernel << <(activeBlocks + 1) / 1024 + 1, 1024 >> >(d_active, d_active_copy, (activeBlocks + 1));
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "copyActiveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				free(h_active);
				free(h_sudokus);
				cudaFree(d_sudokus);
				cudaFree(d_sudokus_target);
				cudaFree(d_active);
				cudaFree(d_active_scan);
				cudaFree(d_active_copy);
				cudaFree(d_active_field);
				return cudaStatus;
			}

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyActiveKernel!\n", cudaStatus);
				free(h_active);
				free(h_sudokus);
				cudaFree(d_sudokus);
				cudaFree(d_sudokus_target);
				cudaFree(d_active);
				cudaFree(d_active_scan);
				cudaFree(d_active_copy);
				cudaFree(d_active_field);
				return cudaStatus;
			}


		cudaFree(d_active);
		cudaStatus = cudaMalloc((void**)&d_active, (activeBlocks + 9 * activeBlocks + 1) * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			printf("Time for the kernel: %f ms\n", time);
			printf("Not enough memory to solve this sudoku!\n");
			//fprintf(stderr, "cudaMalloc failed!");
			free(h_active);
			free(h_sudokus);
			cudaFree(d_sudokus);
			cudaFree(d_sudokus_target);
			cudaFree(d_active);
			cudaFree(d_active_scan);
			cudaFree(d_active_copy);
			cudaFree(d_active_field);
			return cudaStatus;
		}
		
		dev_active_ptr = thrust::device_ptr<int>(d_active);

		activeResetKernel <<<(activeBlocks + 9 * activeBlocks + 1) / 1024 + 1, 1024 >>>(d_active, (activeBlocks + 9 * activeBlocks + 1));
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "activeResetKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				free(h_active);
				free(h_sudokus);
				cudaFree(d_sudokus);
				cudaFree(d_sudokus_target);
				cudaFree(d_active);
				cudaFree(d_active_scan);
				cudaFree(d_active_copy);
				cudaFree(d_active_field);
				return cudaStatus;
			}

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching activeResetKernel!\n", cudaStatus);
				free(h_active);
				free(h_sudokus);
				cudaFree(d_sudokus);
				cudaFree(d_sudokus_target);
				cudaFree(d_active);
				cudaFree(d_active_scan);
				cudaFree(d_active_copy);
				cudaFree(d_active_field);
				return cudaStatus;
			}


		copyKernel << <(activeBlocks + 1) / 1024 + 1, 1024 >> >(d_sudokus, d_sudokus_target, d_active, d_active_scan, d_active_copy, (activeBlocks + 1), newActive);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "copyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				free(h_active);
				free(h_sudokus);
				cudaFree(d_sudokus);
				cudaFree(d_sudokus_target);
				cudaFree(d_active);
				cudaFree(d_active_scan);
				cudaFree(d_active_copy);
				cudaFree(d_active_field);
				return cudaStatus;
			}

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyKernel!\n", cudaStatus);
				free(h_active);
				free(h_sudokus);
				cudaFree(d_sudokus);
				cudaFree(d_sudokus_target);
				cudaFree(d_active);
				cudaFree(d_active_scan);
				cudaFree(d_active_copy);
				cudaFree(d_active_field);
				return cudaStatus;
			}

		cudaFree(d_active_scan);
		cudaStatus = cudaMalloc((void**)&d_active_scan, (activeBlocks + 9 * activeBlocks + 1) * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			printf("Time for the kernel: %f ms\n", time);
			printf("Not enough memory to solve this sudoku!\n");
			//fprintf(stderr, "cudaMalloc failed!");
			free(h_active);
			free(h_sudokus);
			cudaFree(d_sudokus);
			cudaFree(d_sudokus_target);
			cudaFree(d_active);
			cudaFree(d_active_scan);
			cudaFree(d_active_copy);
			cudaFree(d_active_field);
			return cudaStatus;
		}
		
		dev_active_scan_ptr = thrust::device_ptr<int>(d_active_scan);

		cudaFree(d_sudokus);
		cudaFree(d_active_copy);
		d_sudokus = d_sudokus_target;

		activeBlocks = newActive - 1;
	}

	cudaStatus = cudaMemcpy(h_sudokus, d_sudokus, 1 * sizeof(Sudoku), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		free(h_active);
		free(h_sudokus);
		cudaFree(d_sudokus);
		cudaFree(d_active);
		cudaFree(d_active_scan);
		cudaFree(d_active_field);
		return cudaStatus;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f ms\n", time);

	printf("Original:\n");
	PrintSudoku(sudokuArray);
	printf("Solved:\n");
	PrintSudoku(h_sudokus[0].board);

	free(h_active);
	free(h_sudokus);
	cudaFree(d_sudokus);
	cudaFree(d_active);
	cudaFree(d_active_scan);

	return cudaStatus;
}

int main()
{
	byte sudoku[SIZE][SIZE];
	cudaError_t cudaStatus;

	printf("Entry:\n");
	if (ReadSudoku(sudoku, "Entry.txt"))
		return 1;
	cudaStatus = SolveSudoku(sudoku);
	if (cudaStatus != cudaSuccess) {
		printf("Solution not found!\n");
		//fprintf(stderr, "PrepareSudoku failed!");
		return 1;
	}
	getchar();
	printf("------------------------------------------------------------\n");

	printf("Easy:\n");
	if (ReadSudoku(sudoku, "Easy.txt"))
		return 1;
	cudaStatus = SolveSudoku(sudoku);
	if (cudaStatus != cudaSuccess) {
		printf("Solution not found!\n");
		//fprintf(stderr, "PrepareSudoku failed!");
		return 1;
	}
	getchar();
	printf("------------------------------------------------------------\n");

	printf("Easy copy:\n");
	if (ReadSudoku(sudoku, "Easy copy.txt"))
		return 1;
	cudaStatus = SolveSudoku(sudoku);
	if (cudaStatus != cudaSuccess) {
		printf("Solution not found!\n");
		//fprintf(stderr, "PrepareSudoku failed!");
		return 1;
	}
	getchar();
	printf("------------------------------------------------------------\n");

	printf("Medium:\n");
	if (ReadSudoku(sudoku, "Medium.txt"))
		return 1;
	cudaStatus = SolveSudoku(sudoku);
	if (cudaStatus != cudaSuccess) {
		printf("Solution not found!\n");
		//fprintf(stderr, "PrepareSudoku failed!");
		return 1;
	}
	getchar();
	printf("------------------------------------------------------------\n");

	printf("Hard:\n");
	if (ReadSudoku(sudoku, "Hard.txt"))
		return 1;
	cudaStatus = SolveSudoku(sudoku);
	if (cudaStatus != cudaSuccess) {
		printf("Solution not found!\n");
		//fprintf(stderr, "PrepareSudoku failed!");
		return 1;
	}
	getchar();
	printf("------------------------------------------------------------\n");

	printf("Evil:\n");
	if (ReadSudoku(sudoku, "Evil.txt"))
		return 1;
	cudaStatus = SolveSudoku(sudoku);
	if (cudaStatus != cudaSuccess) {
		printf("Solution not found!\n");
		//fprintf(stderr, "PrepareSudoku failed!");
		return 1;
	}
	getchar();
    return 0;
}