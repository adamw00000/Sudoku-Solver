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
	if (resultSet)
		return;

	int emptyCells = 0;

	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			if (board[i][j] == 0)
			{
				emptyCells++;
				//printf("Inserting at (%d, %d)\n", i, j);
				//PrintSudoku(board);
				//getchar();
				//printf("Empty cells: %d\n", emptyCells);

				byte row = i;
				byte col = j;
				byte cellnr = cell(i, j);
				byte availableNumbers = 0;
				for (byte number = 1; number <= SIZE; number++)
				{
					if (!IsNumberInConstraintStructure(number, constraintStructures[row], constraintStructures[col], constraintStructures[cellnr]))
					{
						availableNumbers++;

						board[row][col] = number;
						AddNumberToConstraintStructure(number, constraintStructures[row], constraintStructures[col], constraintStructures[cellnr]);

						printf("Inserting at (%d, %d), number: %d\n", i, j, number);
						PrintSudoku(board);
						getchar();
						Solve();

						RemoveNumberFromRowOrColumn(number, constraintStructures[row], constraintStructures[col], constraintStructures[cellnr]);
						board[row][col] = 0;
					}
				}
				if (availableNumbers == 0)
					return;
			}
		}
	}

	if (emptyCells == 0)
	{
		printf("CPU solved\n");
		SetResult();
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
