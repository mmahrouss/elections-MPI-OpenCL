
# elections-OpenCL
Distributed election program using OpenCL

# Team:
* Mohamed Mahrous 201601078
* Alaa Roshdy 201600031

## Round 1 Code Skeleton

First, the input text file is read line by line and the first vote from each voter is stored in a dynamically located array called `firstVotes`. To ensure that `firstVotes` will always have an array of size multiples of 2, `firstVotes` is padded with -1, making its size the same as the global size. Then, this array is passed to the kernel `getVotes`. `getVotes` creates a one-hot encoding of each of the voter's choice of candidate and stores it in a local array called `localVotes`. To do so, the candidate number voted for by a voter is stored in a variable called `voter`. Then, `localVotes` is initialized with zeros. The size of `localVotes` is C*n where C is the number of candidates and n is the number of work items per work groups (which is initialized in the main program to be 16). Depending on the candidate number and whether or not the vote is just a padding (-1), `localVotes` 's `local_id * C + (vote - 1)` is assigned a 1. This creates a one-hot encoded vector of the voters' fist candidate choice. Then, `localVotes`, the number of candidates and the dynamically allocated `sumVotesOut` is passed to `reduce` kernel. This kernel works in the divide and conquer concept.![reduce kernel illustrated](https://dournac.org/info/images/kernel-code-sum-reduction.png)

For a single work group, we loop over half the work items and sum with a stride of half workgroup size. Then, the final result at `local_id` 0 is written to `sumVotesOut`, making this array have the final sums of all candidates voted for. Percentages are then calculated in the main function.

## Round 2 Code Skeleton

If there is no winner, the top 2 candidates then get filtered to the next round. In the second round, the input file is read line by line and all votes are written in `allVotes` array. This array is then padded with -1 to ensure that the whole array is a multiple of 2 of size number of candidates * globalsize. Then this array is sent in a kernel called `getRoundTwoVotes`. This kernel checks all the voters' candidate choices and assigns 1 to the voter with higher preference. Then, the function reduce that was used in round one is called again but instead of reducing over number of candidates C, it is reducing over the top 2 candidates by summing all the 1s. This results in an an array of size 2 having the reduced sum of the top 2 candidates. In the main function, these 2 values are compared and the candidate with a higher sum is the winner.
