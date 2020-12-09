#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include "mpi.h"

void getTop2(int *votes, int C, int *top2)
{
    top2[0] = top2[1] = -1;
    int max_vote = INT_MIN;
    for (int i = 0; i < C; i++)
    {
        if (votes[i] > max_vote)
        {
            top2[0] = i;
            max_vote = votes[i];
        }
    }
    max_vote = INT_MIN;
    for (int i = 0; i < C; i++)
    {
        if (i == top2[0])
            continue;
        if (votes[i] > max_vote)
        {
            top2[1] = i;
            max_vote = votes[i];
        }
    }
}
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
    fp = fopen("input_file.txt", "r"); // File to read from, should be accessible to all processes

    fscanf(fp, "%d\n", &C); // Read number of Candidates
    fscanf(fp, "%d\n", &V); // Read number of Voters
    fpos_t init_pos;
    fgetpos(fp, &init_pos);
    int my_start = rank * (V / numWorkers);                                  // Vote to start from based on rank
    int my_end = (rank == numWorkers - 1) ? V : my_start + (V / numWorkers); // end
    int *my_votes = (int *)malloc(C * sizeof(int));                          // Array to hold votes sum
    for (int i = 0; i < C; i++)
        my_votes[i] = 0; // Init array to 0

    fseek(fp, getLineChars(C) * my_start, SEEK_CUR); //Seek file to allign with my_start
    int *line = (int *)malloc(C * sizeof(int));      // Array to hold one vote line
    for (int i = my_start; i < my_end; i++)
    {
        readLine(fp, C, line);      // Read one vote
        my_votes[line[0] - 1] += 1; // Register first vote for first round
    }
    // Reduce the votes at root node
    int *sum_votes = (int *)malloc(C * sizeof(int)); // Array to hold total votes
    MPI_Reduce(my_votes, sum_votes, C, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    double *percent_votes = (double *)malloc(C * sizeof(double)); // Array to hold total votes
    int winner = 0;
    if (rank == 0)
    {
        printf("Round 1 results\n============================\n");
        for (int i = 0; i < C; i++)
        {
            percent_votes[i] = sum_votes[i] / (double)V;
            printf("Candidate [%d] got %d/%d which is %0.2lf%%\n", i + 1, sum_votes[i], V, percent_votes[i] * 100);
            if (percent_votes[i] > 0.5)
            {
                winner = i + 1;
            }
        }
        if (winner)
        {
            printf("%d 1\n", winner);
        }
    }
    MPI_Bcast(&winner, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!winner)
    {
        // Round 2
        int top2[2];
        if (rank == 0)
            getTop2(sum_votes, C, top2);
        MPI_Bcast(&top2, 2, MPI_INT, 0, MPI_COMM_WORLD);
        fsetpos(fp, &init_pos);
        fseek(fp, getLineChars(C) * my_start, SEEK_CUR);
        int round_two_votes[2] = {0, 0};
        for (int i = my_start; i < my_end; i++)
        {
            readLine(fp, C, line); // Read one vote
            for (int j = 0; j < C; j++)
            {
                // printf("P: %d, line no: %d vote no: %d is %d\n", rank, i, j, line[j]);
                if ((line[j] - 1) == top2[0])
                {
                    round_two_votes[0] += 1;
                    break;
                }
                if ((line[j] - 1) == top2[1])
                {
                    round_two_votes[1] += 1;
                    break;
                }
            }
        }
        int round_two_sum[2];
        MPI_Reduce(round_two_votes, round_two_sum, 2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            printf("Round 2 results\n============================\n");
            int arg_winner = (round_two_sum[0] > round_two_sum[1]) ? 0 : 1;
            printf("Candidate [%d] got %d/%d which is %0.2lf%%\n", top2[0] + 1, round_two_sum[0], V, (round_two_sum[0] / (double)V) * 100);
            printf("Candidate [%d] got %d/%d which is %0.2lf%%\n", top2[1] + 1, round_two_sum[1], V, (round_two_sum[1] / (double)V) * 100);
            winner = top2[arg_winner] + 1;
            printf("%d 2\n", winner);
        }
    }

    // Cleanup, close file, free memory and finalize MPI
    fclose(fp);
    free(my_votes);
    free(sum_votes);
    free(line);
    MPI_Finalize();
    return 0;
}