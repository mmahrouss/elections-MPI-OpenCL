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
    cl_platform_id cpPlatform;                                                // OpenCL platform
    cl_device_id device_id;                                                   // device ID
    cl_context context;                                                       // context
    cl_command_queue queue;                                                   // command queue
    cl_program program;                                                       // program
    cl_kernel getVotesKernel, iterativeReducerKernel, getRoundtwoVotesKernel; // kernel
    cl_int err;                                                               // Errors
    char *source;                                                             // Source code
    size_t source_size;                                                       // Source code size
    char fileName[] = "kernels.cl";                                           // Kernel source path

    // Host arrays
    int *firstVotes;
    int *sumVotesOut;
    int *sumVotesIn;
    int *allVotes;
    int *sumRoundTwoVotesOut;
    int *sumRoundTwoVotesIn;
    // Device input buffers
    cl_mem d_firstVotes;
    cl_mem d_sumVotesOut;
    cl_mem d_sumVotesIn;
    cl_mem d_sumRoundTwoVotesOut;
    cl_mem d_sumRoundTwoVotesIn;
    cl_mem d_top2;
    cl_mem d_allVotes;

    /* Initalize OpenCL vars end */
    /* File Read start */
    FILE *fp, *kernelFile;
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

    int *line = (int *)malloc(C * sizeof(int));                     // Array to hold one vote line
    firstVotes = (int *)malloc(globalSize * sizeof(int));           // Array to hold total votes
    sumVotesOut = (int *)malloc(nGroups * C * sizeof(int));         // Array to hold total votes
    sumVotesIn = (int *)malloc(nGroups * C * sizeof(int));          // Array to hold sum of votes votes
    sumRoundTwoVotesOut = (int *)malloc(nGroups * 2 * sizeof(int)); //Array to hold sum of votes for the top 2 cands
    sumRoundTwoVotesIn = (int *)malloc(nGroups * 2 * sizeof(int));
    allVotes = (int *)malloc(C * globalSize * sizeof(int));

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
    kernelFile = fopen(fileName, "r");
    if (!kernelFile)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source, 1, MAX_SOURCE_SIZE, kernelFile);
    fclose(kernelFile);

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
    /* Create a buffer for EVERYONE */
    d_firstVotes = clCreateBuffer(context, CL_MEM_READ_ONLY, globalSize * sizeof(int), NULL, NULL);
    d_sumVotesOut = clCreateBuffer(context, CL_MEM_READ_WRITE, nGroups * C * sizeof(int), NULL, NULL);
    d_sumVotesIn = clCreateBuffer(context, CL_MEM_READ_WRITE, nGroups * C * sizeof(int), NULL, NULL);
    d_sumRoundTwoVotesIn = clCreateBuffer(context, CL_MEM_READ_WRITE, nGroups * 2 * sizeof(int), NULL, NULL);
    d_sumRoundTwoVotesOut = clCreateBuffer(context, CL_MEM_READ_WRITE, nGroups * 2 * sizeof(int), NULL, NULL);
    d_allVotes = clCreateBuffer(context, CL_MEM_READ_ONLY, globalSize * C * sizeof(int), NULL, NULL);
    d_top2 = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(int), NULL, NULL);

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
    getRoundtwoVotesKernel = clCreateKernel(program, "getRoundtwoVotes", &err);
    clFinish(queue);
    free(firstVotes);

    int nPrev;
    while (nGroups > 1)
    {
        // printf("============================\n");
        // for (int i = 0; i < C * nGroups; i++)
        // {
        //     if (!(i % C))
        //         printf("\n");
        //     printf("%d ", sumVotesOut[i]);
        // }
        // printf("\n");
        nPrev = nGroups;
        memcpy(sumVotesIn, sumVotesOut, nGroups * C * sizeof(int));
        err = clEnqueueWriteBuffer(queue, d_sumVotesIn, CL_TRUE, 0, nGroups * C * sizeof(int), sumVotesIn, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Failed to write buffer.\n");
            exit(1);
        }

        nGroups = ceil(nGroups / (float)LOCALSIZE);
        globalSize = nGroups * LOCALSIZE;

        // printf("%ld %ld \n", nGroups, globalSize);
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
    // for (int i = 0; i < C; i++)
    // {
    //     printf("%d ", sumVotesOut[i]);
    // }
    // printf("\n");
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
    free(sumVotesIn);
    free(percent_votes);
    clReleaseMemObject(d_sumVotesOut);
    clReleaseMemObject(d_sumVotesIn);
    clReleaseMemObject(d_firstVotes);
    clReleaseKernel(getVotesKernel);
    if (winner)
    {
        printf("%d 1\n", winner);
    }
    else
    {
        int top2[2];
        getTop2(sumVotesOut, C, top2);
        fsetpos(fp, &init_pos);

        // printf("Top 2 candidates = %d and %d", top2[0], top2[1]);

        nGroups = ceil(V / (float)LOCALSIZE);
        globalSize = nGroups * LOCALSIZE;

        err = clEnqueueWriteBuffer(queue, d_top2, CL_TRUE, 0, 2 * sizeof(int), top2, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Failed to enque write buffer. 3.0 \n");
            exit(1);
        }

        for (int i = 0; i < V; i++)
        {
            readLine(fp, C, line); // Read one vote
            for (int j = 0; j < C; j++)
            {
                allVotes[i * C + j] = line[j] - 1;
            }
        }
        for (int i = V * C; i < globalSize * C; i++)
        {
            allVotes[i] = -1; // padding
        }

        err = clEnqueueWriteBuffer(queue, d_allVotes, CL_TRUE, 0, C * globalSize * sizeof(int), allVotes, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Failed to enque write buffer. 3.0 \n");
            exit(1);
        }

        err = clSetKernelArg(getRoundtwoVotesKernel, 0, sizeof(int), (void *)&C);
        err |= clSetKernelArg(getRoundtwoVotesKernel, 1, sizeof(cl_mem), (void *)&d_allVotes);
        err |= clSetKernelArg(getRoundtwoVotesKernel, 2, sizeof(int) * LOCALSIZE * 2, NULL);
        err |= clSetKernelArg(getRoundtwoVotesKernel, 3, sizeof(cl_mem), (void *)&d_top2);
        err |= clSetKernelArg(getRoundtwoVotesKernel, 4, sizeof(cl_mem), (void *)&d_sumRoundTwoVotesOut);
        if (err != CL_SUCCESS)
        {
            printf("Failed to set kernel args getRoundtwoVotesKernel\n");
            printf("Error %d\n", err);
            exit(1);
        }
        err = clEnqueueNDRangeKernel(
            queue,
            getRoundtwoVotesKernel,
            1, NULL,
            &globalSize,
            &localSize,
            0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Failed to enqueue NDrange. 3.0\n");
            printf("Error %d\n", err);
            exit(1);
        }

        err = clEnqueueReadBuffer(queue, d_sumRoundTwoVotesOut, CL_TRUE, 0, nGroups * 2 * sizeof(int), sumRoundTwoVotesOut, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Failed to enqueue read buffer. 3.0 \n");
            exit(1);
        }
        while (nGroups > 1)
        {
            // printf("============================");
            // for (int i = 0; i < 2 * nGroups; i++)
            // {
            //     if (!(i % 2))
            //         printf("\n");
            //     printf("%d ", sumRoundTwoVotesOut[i]);
            // }
            // printf("\n");
            nPrev = nGroups;
            memcpy(sumRoundTwoVotesIn, sumRoundTwoVotesOut, nGroups * 2 * sizeof(int));
            err = clEnqueueWriteBuffer(queue, d_sumRoundTwoVotesIn, CL_TRUE, 0, nGroups * 2 * sizeof(int), sumRoundTwoVotesIn, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Failed to write buffer.\n");
                exit(1);
            }

            nGroups = ceil(nGroups / (float)LOCALSIZE);
            globalSize = nGroups * LOCALSIZE;

            // printf("%ld %ld \n", nGroups, globalSize);
            int r2C = 2;
            err = clSetKernelArg(iterativeReducerKernel, 0, sizeof(int), (void *)&r2C);
            err |= clSetKernelArg(iterativeReducerKernel, 1, sizeof(int), (void *)&nPrev);
            err |= clSetKernelArg(iterativeReducerKernel, 2, sizeof(cl_mem), (void *)&d_sumRoundTwoVotesIn);
            err |= clSetKernelArg(iterativeReducerKernel, 3, sizeof(int) * 2 * LOCALSIZE, NULL);
            err |= clSetKernelArg(iterativeReducerKernel, 4, sizeof(cl_mem), (void *)&d_sumRoundTwoVotesOut);
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
            err = clEnqueueReadBuffer(queue, d_sumRoundTwoVotesOut, CL_TRUE, 0, nGroups * 2 * sizeof(int), sumRoundTwoVotesOut, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Failed to enqueue read buffer. 2.0 \n");
                exit(1);
            }
            clFinish(queue);
        }
        printf("Round 2 results\n============================\n");
        int arg_winner = (sumRoundTwoVotesOut[0] > sumRoundTwoVotesOut[1]) ? 0 : 1;
        printf("Candidate [%d] got %d/%d which is %0.2lf%%\n", top2[0] + 1, sumRoundTwoVotesOut[0], V, (sumRoundTwoVotesOut[0] / (double)V) * 100);
        printf("Candidate [%d] got %d/%d which is %0.2lf%%\n", top2[1] + 1, sumRoundTwoVotesOut[1], V, (sumRoundTwoVotesOut[1] / (double)V) * 100);
        winner = top2[arg_winner] + 1;
        printf("%d 2\n", winner);
    }

    /* free host resources */
    free(line);
    free(source);
    free(sumVotesOut);
    free(sumRoundTwoVotesOut);
    free(sumRoundTwoVotesIn);
    free(allVotes);

    /* free OpenCL resources */
    clReleaseMemObject(d_sumRoundTwoVotesOut);
    clReleaseMemObject(d_sumRoundTwoVotesIn);
    clReleaseMemObject(d_allVotes);
    clReleaseMemObject(d_top2);

    clReleaseCommandQueue(queue);

    clReleaseKernel(iterativeReducerKernel);
    clReleaseKernel(getRoundtwoVotesKernel);

    clReleaseProgram(program);
    clReleaseContext(context);
    // printf("smooth\n");
    return 0;
}