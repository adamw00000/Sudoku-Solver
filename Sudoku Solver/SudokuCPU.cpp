#include "SudokuCPU.h"

SudokuCPU::SudokuCPU(byte board[SIZE][SIZE], uint32_t constraintStructures[SIZE])
{
	for (int i = 0; i < SIZE; i++)
	{
		this->constraintStructures[i] = constraintStructures[i];
		for (int j = 0; j < SIZE; j++)
		{
			this->board[i][j] = board[i][j];
		}
	}
}

void SudokuCPU::GetEmptyFields()
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

void SudokuCPU::SetResult()
{
	resultSet = true;
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			result[i][j] = board[i][j];
		}
	}
}

byte SudokuCPU::cell(byte i, byte j)
{
	return (i / 3) * 3 + j / 3;
}

void SudokuCPU::PrintSudoku(byte arr[SIZE][SIZE])
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

void SudokuCPU::Solve()
{
	Solve(0);
}
void SudokuCPU::Solve(int i)
{
	if (i == SIZE * SIZE || emptyFields[i] == (byte)-1)
	{
		printf("CPU solution found!\n");
		SetResult();
		return;
	}
	if (resultSet)
		return;

	int field = emptyFields[i];
	byte row = field / SIZE;
	byte col = field % SIZE;
	
	byte cellnr = cell(row, col);
	
	if (board[row][col] == 0)
	{
		for (byte number = 1; number <= SIZE; number++)
		{
			if (!IsNumberInConstraintStructure(number, constraintStructures[row], constraintStructures[col], constraintStructures[cellnr]))
			{
				board[row][col] = number;
				AddNumberToConstraintStructure(number, constraintStructures[row], constraintStructures[col], constraintStructures[cellnr]);

				i++;
				Solve(i);
				i--;

				RemoveNumberFromRowOrColumn(number, constraintStructures[row], constraintStructures[col], constraintStructures[cellnr]);
				board[row][col] = 0;
			}
		}
	}
}

bool SudokuCPU::IsNumberInConstraintStructure(byte number, const uint32_t& rowStructure, const uint32_t& colStructure, const uint32_t& cellStructure)
{
	return (rowStructure & (1U << number) || colStructure & (1U << (SIZE + number)) || cellStructure & (1U << (SIZE + SIZE + number)));
}

void SudokuCPU::AddNumberToConstraintStructure(byte number, uint32_t& rowStructure, uint32_t& colStructure, uint32_t& cellStructure)
{
	if (number != 0)
	{
		rowStructure |= (1U << number);
		colStructure |= (1U << (SIZE + number));
		cellStructure |= (1U << (SIZE + SIZE + number));
	}
}

void SudokuCPU::RemoveNumberFromRowOrColumn(byte number, uint32_t& rowStructure, uint32_t& colStructure, uint32_t& cellStructure)
{
	if (number != 0)
	{
		rowStructure &= ~(1U << number);
		colStructure &= ~(1U << (SIZE + number));
		cellStructure &= ~(1U << (SIZE + SIZE + number));
	}
}
