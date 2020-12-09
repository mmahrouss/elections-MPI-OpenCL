#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

int getLineChars(int c)
{
    int power = 10;
    int result = c * 2;
    while (c >= power)
    {
        result += (c - power + 1);
        power *= 10;
    }
    return result;
}
void readLine(FILE *fp, int C, int *out)
{
    for (int i = 0; i < C - 1; i++)
    {
        fscanf(fp, "%d ", &out[i]);
    }
    fscanf(fp, "%d\n", &out[C - 1]);
}
int main(int argc, char **argv)
{
    FILE *fp;
    int rank, numWorkers;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numWorkers);

    int C, V;
    fp = fopen("input.txt", "r");

    fscanf(fp, "%d\n", &C);
    fscanf(fp, "%d\n", &V);

    int my_start = rank * (V / numWorkers);
    int my_end = rank == numWorkers - 1 ? V : my_start + (V / numWorkers);
    int *my_votes = (int *)malloc(C * sizeof(int));
    for (int i = 0; i < C; i++)
        my_votes[i] = 0;

    fseek(fp, getLineChars(C) * my_start, SEEK_CUR);
    int *line = (int *)malloc(C * sizeof(int));
    for (; my_start < my_end; my_start++)
    {
        readLine(fp, C, line);
        my_votes[line[0] - 1] += 1;
    }

    fclose(fp);
    free(my_votes);
    free(line);
    MPI_Finalize();
    return 0;
}