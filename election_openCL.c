#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <CL/cl.h>

#define LOCALSIZE 16
#define MAX_SOURCE_SIZE (0x100000)

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
    cl_platform_id cpPlatform;      // OpenCL platform
    cl_device_id device_id;         // device ID
    cl_context context;             // context
    cl_command_queue queue;         // command queue
    cl_program program;             // program
    cl_kernel getVotesKernel;       // kernel
    cl_int err;                     // Errors
    char *source;                   // Source code
    size_t source_size;             // Source code size
    char fileName[] = "kernels.cl"; // Kernel source path

    // Host arrays
    int *firstVotes;
    int *allVotes;
    // Device input buffers
    cl_mem d_firstVotes;
    cl_mem d_allVotes;

    /* Initalize OpenCL vars end */
    /* File Read start */
    FILE *fp;
    int rank, numWorkers;
    int N, C, V;
    fp = fopen("input_file.txt", "r"); // File to read from, should be accessible to all processes

    fscanf(fp, "%d\n", &C); // Read number of Candidates
    fscanf(fp, "%d\n", &V); // Read number of Voters
    fpos_t init_pos;
    fgetpos(fp, &init_pos);

    N = ceil(V / (float)LOCALSIZE) * LOCALSIZE;

    int *line = (int *)malloc(C * sizeof(int));    // Array to hold one vote line
    firstVotes = (int *)malloc(N * sizeof(int));   // Array to hold total votes
    allVotes = (int *)malloc(N * C * sizeof(int)); // Array to hold total votes
    for (int i = 0; i < V; i++)
    {
        readLine(fp, C, line);   // Read one vote
        firstVotes[i] = line[0]; // Register first vote for first round
    }
    for (int i = V; i < N; i++)
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

    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);

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
    d_firstVotes = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, NULL);
    d_allVotes = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * C * sizeof(int), NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_firstVotes, CL_TRUE, 0, N * sizeof(int), firstVotes, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Failed to write buffer.\n");
        exit(1);
    }
    clSetKernelArg(getVotesKernel, 0, sizeof(int), (void *)&C);
    clSetKernelArg(getVotesKernel, 1, sizeof(cl_mem), (void *)&d_firstVotes);
    clSetKernelArg(getVotesKernel, 2, sizeof(int) * LOCALSIZE, NULL);
    clSetKernelArg(getVotesKernel, 3, sizeof(cl_mem), (void *)&d_allVotes);

    int local = LOCALSIZE;
    err = clEnqueueNDRangeKernel(
        queue,
        getVotesKernel,
        1, NULL,
        &N,
        &local,
        0, NULL, NULL);
    err = clEnqueueReadBuffer(queue, d_allVotes, CL_TRUE, 0, N * C * sizeof(int), allVotes, 0, NULL, NULL);
    clFinish(queue);

    /* free host resources */
    free(line);
    free(firstVotes);
    free(source);

    /* free OpenCL resources */
    clReleaseMemObject(d_allVotes);
    clReleaseMemObject(d_firstVotes);
    clReleaseCommandQueue(queue);
    clReleaseKernel(getVotesKernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}