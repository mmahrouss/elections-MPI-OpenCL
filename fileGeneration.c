#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

void write_to_file(int* recvbuf, int cands, int voters){
    FILE* fp = fopen("input_file.txt", "w");
    fprintf(fp, "%d\n", cands); //no of candidates
    fprintf(fp, "%d\n", voters); //no of voters

    //Writes numbers to File
    for (int i = 0; i < voters; i++)
    {
        for (int j = 0; j < cands; j++)
            fprintf(fp, "%d ", *(recvbuf + i*cands + j));
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void shuffle(int *array, int size)
{
    if (size > 1) 
    {
        for (int i = 0; i < size - 1; i++) 
        {
          int j = i + rand() / (RAND_MAX / (size - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

int main(int argc, char *argv[]){
    int rank, numOfProcesses, x, result, chunk, excess_from_chunk;
    int voters, cands;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);

    // scanf the number of voters and number of candidates
    if(rank == 0)
    {
        printf("Please enter number of candidates: ");
        scanf("%d",  & cands); //number of columns
        printf("Please enter number of voters: ");
        scanf("%d",  & voters); //number of rows 
    }
    MPI_Bcast(&voters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cands, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform the operation u need on each node
    if(rank==numOfProcesses-1){
        chunk = (voters/numOfProcesses) + (voters % numOfProcesses);
    }
    else
    {
        chunk = (voters/numOfProcesses);
    }
    /* Intializes random number generator with different seeds depending on the rank */
    srand(time(0)+rank);
    int* sendbuf =(int*) malloc ( chunk*cands * sizeof(int) );
    int* votes = (int*) malloc ( cands * sizeof(int) );
    for(int i=0;i<cands;i++){
        votes[i]=i+1;
    }
    
    // Generate the array of random numbers
    for (int i = 0; i <  chunk; i++){
        shuffle(votes,cands);
        for (int j = 0; j < cands; j++){
        // Random number from 1 to cands
         sendbuf[i*cands + j] = votes[j];
        }
    }

    // Gather all the info in recvbuf
    int *recvbuf = (int *)malloc(voters * cands * sizeof(int));
    int* recvcnts =(int*) malloc ( numOfProcesses * sizeof(int) );
    int* dpsl =(int*) malloc ( numOfProcesses * sizeof(int) );

    for(int i=0; i<numOfProcesses;i++){
        if(i==numOfProcesses-1){
            recvcnts[i] = ((voters/numOfProcesses) + (voters % numOfProcesses))*cands;
        }
        else
        {
            recvcnts[i] = (voters/numOfProcesses)*cands;
        }
        dpsl[i] = i*(voters/numOfProcesses)*cands;
    }
    MPI_Gatherv(sendbuf, recvcnts[rank], MPI_INT, recvbuf, recvcnts, dpsl, MPI_INT, 0, MPI_COMM_WORLD);
    // Receive buffer should countain a pointer to a 2D array containing all voter choices
    // write to file what you have gathered
    if(rank == 0)
    {
        write_to_file(recvbuf,cands,voters);
    }

    MPI_Finalize();
    return 0;
} 
