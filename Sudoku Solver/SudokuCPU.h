#pragma once

#include <thrust/extrema.h>

#define SIZE 9
typedef uint8_t byte;

class SudokuCPU
{
	bool resultSet = false;
	byte board[SIZE][SIZE];
	byte emptyFields[SIZE * SIZE];
	uint32_t constraintStructures[SIZE];

	byte cell(byte i, byte j);
	void GetEmptyFields();
	void PrintSudoku(byte arr[SIZE][SIZE]);

	void Solve(int);
	bool IsNumberInConstraintStructure(byte number, const uint32_t & rowStructure, const uint32_t & colStructure, const uint32_t & cellStructure);
	void AddNumberToConstraintStructure(byte number, uint32_t & rowStructure, uint32_t & colStructure, uint32_t & cellStructure);
	void RemoveNumberFromRowOrColumn(byte number, uint32_t & rowStructure, uint32_t & colStructure, uint32_t & cellStructure);
	
	void SetResult();
public:
	byte result[SIZE][SIZE];

	SudokuCPU(byte[SIZE][SIZE], uint32_t[SIZE]);
	void Solve();
};

