#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <CL/cl.h>
#include <math.h>

#define LOCALSIZE 16
#define MAX_SOURCE_SIZE (0x100000)

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
    // Calculates the number of digits and whitespace in the sequence
    /*1 2 3 .... V\n*/
    int power = 10;
    int result = c * 2; // single digits and whitespaces
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
    /* Initalize OpenCL vars start */
    cl_platform_id cpPlatform;                        // OpenCL platform
    cl_device_id device_id;                           // device ID
    cl_context context;                               // context
    cl_command_queue queue;                           // command queue
    cl_program program;                               // program
    cl_kernel getVotesKernel, iterativeReducerKernel; // kernel
    cl_int err;                                       // Errors
    char *source;                                     // Source code
    size_t source_size;                               // Source code size
    char fileName[] = "kernels.cl";                   // Kernel source path

    // Host arrays
    int *firstVotes;
    int *sumVotesOut;
    int *sumVotesIn;
    // Device input buffers
    cl_mem d_firstVotes;
    cl_mem d_sumVotesOut;
    cl_mem d_sumVotesIn;

    /* Initalize OpenCL vars end */
    /* File Read start */
    FILE *fp;
    int rank, numWorkers;
    // V is number of votes, C is number of candidates, N is number of work groups
    size_t globalSize, nGroups;
    int C, V;
    fp = fopen("input_file.txt", "r"); // File to read from, should be accessible to all processes

    fscanf(fp, "%d\n", &C); // Read number of Candidates
    fscanf(fp, "%d\n", &V); // Read number of Voters
    fpos_t init_pos;
    fgetpos(fp, &init_pos);

    nGroups = ceil(V / (float)LOCALSIZE);
    globalSize = nGroups * LOCALSIZE;

    int *line = (int *)malloc(C * sizeof(int));             // Array to hold one vote line
    firstVotes = (int *)malloc(globalSize * sizeof(int));   // Array to hold total votes
    sumVotesOut = (int *)malloc(nGroups * C * sizeof(int)); // Array to hold total votes
    sumVotesIn = (int *)malloc(nGroups * C * sizeof(int));  // Array to hold sum of votes votes
    for (int i = 0; i < V; i++)
    {
        readLine(fp, C, line);   // Read one vote
        firstVotes[i] = line[0]; // Register first vote for first round
    }
    for (int i = V; i < globalSize; i++)
    {
        firstVotes[i] = -1; // padding
    }
    /* File Read End */

    /* OpenCL Setup start */

    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Failed to get platform ID.\n");
        exit(1);
    }
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Failed to get device id.\n");
        exit(1);
    }
    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Failed to create context.\n");
        exit(1);
    }
    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);

    if (err != CL_SUCCESS)
    {
        printf("Failed to create command queue.\n");
        exit(1);
    }
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                                        (const char **)&source, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Failed to create program from source.\n");
        exit(1);
    }
    /* Build Kernel Program */
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Failed to build program.\n");
        exit(1);
    }
    /* Create OpenCL Kernel */
    getVotesKernel = clCreateKernel(program, "getVotes", &err);
    if (err != CL_SUCCESS)
    {
        printf("Failed to create kernel.\n");
        exit(1);
    }
    /* Create a buffer for the firstVotes */
    d_firstVotes = clCreateBuffer(context, CL_MEM_READ_ONLY, globalSize * sizeof(int), NULL, NULL);
    d_sumVotesOut = clCreateBuffer(context, CL_MEM_READ_WRITE, nGroups * C * sizeof(int), NULL, NULL);
    d_sumVotesIn = clCreateBuffer(context, CL_MEM_READ_WRITE, nGroups * C * sizeof(int), NULL, NULL);

    err = clEnqueueWriteBuffer(queue, d_firstVotes, CL_TRUE, 0, globalSize * sizeof(int), firstVotes, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Failed to write buffer.\n");
        exit(1);
    }
    err = clSetKernelArg(getVotesKernel, 0, sizeof(int), (void *)&C);
    err |= clSetKernelArg(getVotesKernel, 1, sizeof(cl_mem), (void *)&d_firstVotes);
    err |= clSetKernelArg(getVotesKernel, 2, sizeof(int) * C * LOCALSIZE, NULL);
    err |= clSetKernelArg(getVotesKernel, 3, sizeof(cl_mem), (void *)&d_sumVotesOut);
    if (err != CL_SUCCESS)
    {
        printf("Failed to create kernel arg.\n");
        exit(1);
    }
    size_t localSize = LOCALSIZE;
    err = clEnqueueNDRangeKernel(
        queue,
        getVotesKernel,
        1, NULL,
        &globalSize,
        &localSize,
        0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Failed to enqueue NDrange.\n");
        exit(1);
    }
    err = clEnqueueReadBuffer(queue, d_sumVotesOut, CL_TRUE, 0, nGroups * C * sizeof(int), sumVotesOut, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Failed to enqueue read buffer.\n");
        exit(1);
    }
    iterativeReducerKernel = clCreateKernel(program, "iterativeReducer", &err);
    int nPrev;
    while (nGroups > 1)
    {
        printf("============================\n");
        nPrev = nGroups;
        for (int i = 0; i < C * nGroups; i++)
        {
            if (!(i % C))
                printf("\n");
            printf("%d ", sumVotesOut[i]);
        }
        printf("\n");
        memcpy(sumVotesIn, sumVotesOut, nGroups * C * sizeof(int));
        err = clEnqueueWriteBuffer(queue, d_sumVotesIn, CL_TRUE, 0, nGroups * C * sizeof(int), sumVotesIn, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Failed to write buffer.\n");
            exit(1);
        }

        nGroups = ceil(nGroups / (float)LOCALSIZE);
        globalSize = nGroups * LOCALSIZE;

        printf("%ld %ld \n", nGroups, globalSize);
        err = clSetKernelArg(iterativeReducerKernel, 0, sizeof(int), (void *)&C);
        err = clSetKernelArg(iterativeReducerKernel, 1, sizeof(int), (void *)&nPrev);
        err |= clSetKernelArg(iterativeReducerKernel, 2, sizeof(cl_mem), (void *)&d_sumVotesIn);
        err |= clSetKernelArg(iterativeReducerKernel, 3, sizeof(int) * C * LOCALSIZE, NULL);
        err |= clSetKernelArg(iterativeReducerKernel, 4, sizeof(cl_mem), (void *)&d_sumVotesOut);
        if (err != CL_SUCCESS)
        {
            printf("Failed to create kernel arg. 2.0 \n");
            exit(1);
        }

        err = clEnqueueNDRangeKernel(
            queue,
            iterativeReducerKernel,
            1, NULL,
            &globalSize,
            &localSize,
            0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Failed to enqueue NDrange.\n");
            exit(1);
        }
        err = clEnqueueReadBuffer(queue, d_sumVotesOut, CL_TRUE, 0, nGroups * C * sizeof(int), sumVotesOut, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Failed to enqueue read buffer. 2.0 \n");
            exit(1);
        }
    }

    clFinish(queue);
    for (int i = 0; i < C; i++)
    {
        printf("%d ", sumVotesOut[i]);
    }
    printf("\n");
    double *percent_votes = (double *)malloc(C * sizeof(double)); // Array to hold total votes
    int winner = 0;
    printf("Round 1 results\n============================\n");
    for (int i = 0; i < C; i++)
    {
        percent_votes[i] = sumVotesOut[i] / (double)V;
        printf("Candidate [%d] got %d/%d which is %0.2lf%%\n", i + 1, sumVotesOut[i], V, percent_votes[i] * 100);
        if (percent_votes[i] > 0.5)
        {
            winner = i + 1;
        }
    }
    if (winner)
    {
        printf("%d 1\n", winner);
    }
    else
    {
        int top2[2];
        getTop2(sumVotesOut, C, top2);
        fsetpos(fp, &init_pos);
    }

    /* free host resources */
    free(line);
    free(firstVotes);
    free(source);

    /* free OpenCL resources */
    clReleaseMemObject(d_sumVotesOut);
    clReleaseMemObject(d_firstVotes);
    clReleaseCommandQueue(queue);
    clReleaseKernel(getVotesKernel);
    clReleaseKernel(iterativeReducerKernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    printf("smooth\n");
    return 0;
}