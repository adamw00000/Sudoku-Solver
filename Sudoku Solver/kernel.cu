
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>


#define SIZE 9

typedef uint8_t byte;

struct Sudoku
{
	byte board[SIZE][SIZE];
	uint16_t rowNumbers[SIZE];
	uint16_t colNumbers[SIZE];
	uint16_t cellNumbers[SIZE];
	byte rowCounts[SIZE];
	byte colCounts[SIZE];
	byte cellCounts[SIZE];

	//bool active;

	Sudoku(byte board[SIZE][SIZE], uint16_t rowNumbers[SIZE], uint16_t colNumbers[SIZE], 
		uint16_t cellNumbers[SIZE], byte rowCounts[SIZE], byte colCounts[SIZE], byte cellCounts[SIZE])//,bool active)
	{
		//this->active = active;

		for (int i = 0; i < SIZE; i++)
		{
			this->rowNumbers[i] = rowNumbers[i];
			this->colNumbers[i] = colNumbers[i];
			this->cellNumbers[i] = cellNumbers[i];
			this->rowCounts[i] = rowCounts[i];
			this->colCounts[i] = colCounts[i];
			this->cellCounts[i] = cellCounts[i];
			for (int j = 0; j < SIZE; j++)
				this->board[i][j] = board[i][j];
		}
	}
};

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int Size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__host__ __device__ void PrintSudoku(byte arr[SIZE][SIZE])
{
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			printf("%d ", (int)arr[i][j]);
			if (j == SIZE - 1)
				printf("\n");
		}
	}
	printf("\n");
}

__host__ __device__ byte GetBestCount(byte structure[])
{
	char max = -1;
	char index = -1;

	for (byte i = 0; i < SIZE; i++)
	{
		if (structure[i] > max && structure[i] < 9) 
		{
			max = structure[i];
			index = i;
		}
	}

	return (byte)index;
}

__host__ __device__ byte GetBestCountInRow(byte board[SIZE][SIZE], byte columnCounts[], byte row)
{
	char max = -1;
	char index = -1;

	for (byte j = 0; j < SIZE; j++)
	{
		if (board[row][j] == 0 && columnCounts[j] > max)
		{
			max = (char)columnCounts[j];
			index = j;
		}
	}

	return (byte)index;
}

__host__ __device__ bool IsNumberInRowOrColumn(uint16_t structure, byte number)
{
	return structure & (1U << number);
}

__host__ __device__ void AddNumberToRowOrColumn(uint16_t& structure, byte number)
{
	if (number != 0)
	{
		structure |= (1U << number);
	}
}

__host__ __device__ void RemoveNumberFromRowOrColumn(uint16_t& structure, byte number)
{
	if (number != 0)
	{
		structure &= ~(1U << number);
	}
}

__host__ __device__ void PrintRowsOrColumns(uint16_t structure[])
{
	for (byte i = 0; i < SIZE; i++)
	{
		for (byte number = 1; number <= SIZE; number++)
		{
			printf("%d ", (int)IsNumberInRowOrColumn(structure[i], number));
			if (number % (SIZE + 1) == SIZE)
				printf("\n");
		}
	}
	printf("\n");
}

__host__ __device__ byte cell(byte i, byte j)
{
	return (i / 3) * 3 + j / 3;
}

__global__ void activeResetKernel(bool* d_active)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i == 0)
		return;

	d_active[i] = false;
}


__global__ void copyKernel(Sudoku* d_sudokus, Sudoku* d_sudokus_target, bool* d_active, int* d_active_scan, int n, int newMax)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i == 0)
		return;

	if (i < n - 1 && d_active_scan[i] != d_active_scan[i + 1])
	{
		d_sudokus_target[d_active_scan[i]] = d_sudokus[i];

		if (d_active_scan[i] != newMax)
			d_active[d_active_scan[i]] = 1;
	}
}

__global__ void sudokuKernel(Sudoku* d_sudokus, bool* d_active, int n)
{
	//while(1) {

	int i = (blockIdx.x * blockDim.x) +  threadIdx.x;

	if (d_active[0] == true)
	{
		return;
	}

	Sudoku mySudoku = d_sudokus[i];
	if (d_active[i] == false)
	{
		//continue; 
		return;
	}
	//printf("Id: %d\n", i);

	//if (i == 1)
	//	PrintSudoku(mySudoku.board);
	//PrintRowsOrColumns(mySudoku.rowNumbers);
	//return;
	byte row = GetBestCount(mySudoku.rowCounts);
	if (row == (byte)-1) 
	{
		//koniec
		d_sudokus[0] = mySudoku;
		d_active[0] = true;
		return;
	}
	byte col = GetBestCountInRow(mySudoku.board, mySudoku.colCounts, row);
	byte cellnr = cell(row, col);
	for (byte number = 1; number <= SIZE; number++)
	{
		if (!IsNumberInRowOrColumn(mySudoku.rowNumbers[row], number) &&
			!IsNumberInRowOrColumn(mySudoku.colNumbers[col], number) &&
			!IsNumberInRowOrColumn(mySudoku.cellNumbers[cellnr], number))
		{
			mySudoku.board[row][col] = number;

			AddNumberToRowOrColumn(mySudoku.rowNumbers[row], number);
			AddNumberToRowOrColumn(mySudoku.colNumbers[col], number);
			AddNumberToRowOrColumn(mySudoku.cellNumbers[cellnr], number);
			mySudoku.rowCounts[row]++;
			mySudoku.colCounts[col]++;
			mySudoku.cellCounts[cellnr]++;

			//PrintSudoku(d_sudokus[i].board);
			// int size = d_stack[n];
			// int index = d_stack[size - 1];
			// d_stack[n]--;

			int index = n + 1 + (i - 1) * 9 + (number - 1);
			//printf("Tid:%d, (%d, %d), number %d, activates tid %d\n", i, (int)row, (int)col, (int)number, index);
			d_sudokus[index] = mySudoku;
			d_active[index] = true;
			//printf("%d - active? %d\n", index, d_sudokus[index].active);
			RemoveNumberFromRowOrColumn(mySudoku.rowNumbers[row], number);
			RemoveNumberFromRowOrColumn(mySudoku.colNumbers[col], number);
			RemoveNumberFromRowOrColumn(mySudoku.cellNumbers[cellnr], number);
			mySudoku.rowCounts[row]--;
			mySudoku.colCounts[col]--;
			mySudoku.cellCounts[cellnr]--;
			mySudoku.board[row][col] = 0;
		}
	}
	d_active[i] = false;
	//}
}

void ReadSudoku(byte arr[SIZE][SIZE], std::string filename)
{
	std::ifstream stream(filename);

	char c = stream.get();
	byte i = 0, j = 0;

	while (stream.good() && c != '\n') 
	{
		if (c != 'x')
		{
			int n = atoi(&c);
			arr[i][j] = n;
		}
		else
			arr[i][j] = 0;
		c = stream.get();
		j++;
		if (j == SIZE)
		{
			j = 0;
			i++;
		}
	}
	stream.close();
}

void GetRowColNumbers(byte sudoku[SIZE][SIZE], uint16_t rows[], uint16_t columns[], uint16_t areas[])
{
	for (byte i = 0; i < SIZE; i++)
	{
		rows[i] = 0;
		columns[i] = 0;
		areas[i] = 0;
	}

	for (byte i = 0; i < SIZE; i++)
	{
		for (byte j = 0; j < SIZE; j++)
		{
			if (sudoku[i][j] != 0)
			{
				AddNumberToRowOrColumn(rows[i], sudoku[i][j]);
				AddNumberToRowOrColumn(columns[j], sudoku[i][j]);
				AddNumberToRowOrColumn(areas[cell(i, j)], sudoku[i][j]);
			}
		}
	}
}

void GetRowColCounts(uint16_t rows[], uint16_t columns[], uint16_t cells[], byte rowCounts[], byte columnCounts[], byte cellCounts[])
{
	for (byte i = 0; i < SIZE; i++)
	{
		rowCounts[i] = 0;
		columnCounts[i] = 0;
		cellCounts[i] = 0;
	}

	for (byte i = 0; i < SIZE; i++)
	{
		for (byte number = 1; number <= SIZE; number++)
		{
			if (IsNumberInRowOrColumn(rows[i], number))
				rowCounts[i]++;
			if (IsNumberInRowOrColumn(columns[i], number))
				columnCounts[i]++;
			if (IsNumberInRowOrColumn(cells[i], number))
				cellCounts[i]++;
		}
	}
}

cudaError_t PrepareSudoku(byte sudokuArray[SIZE][SIZE])
{
	PrintSudoku(sudokuArray);
	printf("------------------------------------------------------------------------------------\n");

	uint16_t rowNumbers[SIZE];
	uint16_t colNumbers[SIZE];
	uint16_t cellNumbers[SIZE];	
	byte rowCounts[SIZE];
	byte colCounts[SIZE];
	byte cellCounts[SIZE];

	GetRowColNumbers(sudokuArray, rowNumbers, colNumbers, cellNumbers);
	GetRowColCounts(rowNumbers, colNumbers, cellNumbers, rowCounts, colCounts, cellCounts);

	//PrintRowsOrColumns(rowNumbers);
	//PrintRowsOrColumns(colNumbers);
	//PrintRowsOrColumns(cellNumbers);

	cudaError_t cudaStatus;

	Sudoku activeSudoku(sudokuArray, rowNumbers, colNumbers, cellNumbers, rowCounts, colCounts, cellCounts);//, true);
	//Sudoku inactiveSudoku(sudokuArray, rowNumbers, colNumbers, cellNumbers, rowCounts, colCounts, cellCounts);//, false);

	int nBlocks = 1000000;
	int maxActiveBlocks = 1;
	Sudoku *d_sudokus;
	Sudoku *d_sudokus_target;

	Sudoku *h_sudokus = (Sudoku*)malloc(nBlocks * sizeof(Sudoku));
	if (h_sudokus == NULL) {
		fprintf(stderr, "malloc failed!");
		return cudaStatus;
	}
	bool *d_active;
	int *d_active_scan;

	bool *h_active = (bool*)malloc(nBlocks * sizeof(bool));
	if (h_active == NULL) {
		fprintf(stderr, "malloc failed!");
		return cudaStatus;
	}

	for (int i = 0; i < nBlocks; i++) {
		if (i == 1)
		{
			h_active[i] = true;
			h_sudokus[i] = activeSudoku;
		}
		else
		{
			h_active[i] = 0;
		}
	}

	cudaStatus = cudaMalloc((void**)&d_sudokus, nBlocks * sizeof(Sudoku));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(d_sudokus);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(d_sudokus, h_sudokus, nBlocks * sizeof(Sudoku), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(d_sudokus);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&d_active, nBlocks * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(d_sudokus);
		cudaFree(d_active);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&d_active_scan, nBlocks * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(d_sudokus);
		cudaFree(d_active);
		cudaFree(d_active_scan);
		return cudaStatus;
	}


	cudaStatus = cudaMemcpy(d_active, h_active, nBlocks * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(d_sudokus);
		cudaFree(d_active);
		cudaFree(d_active_scan);
		return cudaStatus;
	}
	// Launch a kernel on the GPU with one thread for each element.

	thrust::device_ptr<bool> dev_active_ptr(d_active);
	thrust::device_ptr<int> dev_active_scan_ptr(d_active_scan);
	thrust::device_ptr<Sudoku> dev_sudokus_ptr(d_sudokus);

	int i = 0;
	while (1)
	//for (int i = 0; i < 3; i++)
	{
		i++;
		printf("Iteration: %d\n", i);
		sudokuKernel <<<nBlocks/1024 + 1, 1024>>>(d_sudokus, d_active, maxActiveBlocks);
		cudaDeviceSynchronize();

		if (dev_active_ptr[0] == true)
		{
			printf("Solution found!\n");
			break;
		}

		dev_active_ptr[0] = true;
		thrust::exclusive_scan(dev_active_ptr, dev_active_ptr + nBlocks, dev_active_scan_ptr);
		dev_active_ptr[0] = false;
		thrust::device_ptr<int> newMax = thrust::max_element(dev_active_scan_ptr, dev_active_scan_ptr + nBlocks);

		maxActiveBlocks = maxActiveBlocks + maxActiveBlocks * 9;

		cudaStatus = cudaMalloc((void**)&d_sudokus_target, nBlocks * sizeof(Sudoku));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			cudaFree(d_sudokus);
			return cudaStatus;
		}

		activeResetKernel <<<nBlocks / 1024 + 1, 1024 >>>(d_active);
		cudaDeviceSynchronize();
		copyKernel <<<nBlocks / 1024 + 1, 1024 >>>(d_sudokus, d_sudokus_target, d_active, d_active_scan, nBlocks, *newMax);
		cudaDeviceSynchronize();

		cudaFree(d_sudokus);
		d_sudokus = d_sudokus_target;

		maxActiveBlocks = (*newMax) - 1;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(d_sudokus);
		cudaFree(d_active);
		cudaFree(d_active_scan);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sudokuKernel!\n", cudaStatus);
		cudaFree(d_sudokus);
		cudaFree(d_active);
		cudaFree(d_active_scan);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(h_sudokus, d_sudokus, 1 * sizeof(Sudoku), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(d_sudokus);
		return cudaStatus;
	}

	printf("Original:\n");
	PrintSudoku(sudokuArray);
	printf("Solved:\n");
	PrintSudoku(h_sudokus[0].board);

	cudaFree(d_sudokus);
	cudaFree(d_active);
	cudaFree(d_active_scan);
	return cudaStatus;
}

int main()
{
	byte sudoku[SIZE][SIZE];

	ReadSudoku(sudoku, "Entry.txt");
	cudaError_t cudaStatus = PrepareSudoku(sudoku);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "PrepareSudoku failed!");
		return 1;
	}

    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int Size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, SIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, SIZE>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
