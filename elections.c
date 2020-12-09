#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

int getLineChars(int c)
{
    // Calculated the number of digits and whitespace in the sequence
    /*1 2 3 .... V\n*/
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
    // Reads one voter's votes line
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
    fp = fopen("input.txt", "r"); // File to read from, should be accessible to all processes

    fscanf(fp, "%d\n", &C); // Read number of Candidates
    fscanf(fp, "%d\n", &V); // Read number of Voters

    int my_start = rank * (V / numWorkers);                                // Vote to start from based on rank
    int my_end = rank == numWorkers - 1 ? V : my_start + (V / numWorkers); // end
    int *my_votes = (int *)malloc(C * sizeof(int));                        // Array to hold votes sum
    for (int i = 0; i < C; i++)
        my_votes[i] = 0; // Init array to 0

    fseek(fp, getLineChars(C) * my_start, SEEK_CUR); //Seek file to allign with my_start
    int *line = (int *)malloc(C * sizeof(int));      // Array to hold one vote line
    for (; my_start < my_end; my_start++)
    {
        readLine(fp, C, line);      // Read one vote
        my_votes[line[0] - 1] += 1; // Register first vote for first round
    }

    // Cleanup, close file, free memory and finalize MPI
    fclose(fp);
    free(my_votes);
    free(line);
    MPI_Finalize();
    return 0;
}