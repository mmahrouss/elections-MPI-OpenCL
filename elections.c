#include <stdio.h>

int getLineLength(int c)
{
    int power = 10;
    int result = c * 2;
    while (c > power)
    {
        result += (c - power + 1);
        power *= 10;
    }
    return result;
}
int main(int argc, char **argv)
{
    return 0;
}