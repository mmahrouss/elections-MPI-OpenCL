#include <string.h>
#include <stdlib.h>
#include <stdio.h>

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
    int C, V;
    fp = fopen("input.txt", "r");

    fscanf(fp, "%d\n", &C);
    fscanf(fp, "%d\n", &V);
    int *line = (int *)malloc(C * sizeof(int));

    // get 4rd line
    fseek(fp, getLineChars(C) * 3, SEEK_CUR);
    readLine(fp, C, line);
    for (int i = 0; i < C; i++)
        printf("%d ", line[i]);
    fclose(fp);
    return 0;
}