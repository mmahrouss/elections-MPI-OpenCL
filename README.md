# elections-MPI
Distributed election program using MPI

# Team:
* Mohamed Kasem 201601144
* Mohamed Mahrous 201601078
* Alaa Roshdy 201600031
* Mohamed Mostafa 201600236

## Task 1 input generation
* First, at rank zero, the user is prompted to enter the number of candidates and voters.
* These values are then broadcasted to all other processes. 
* Then, all processes except the last take equal portions of the voters' size calculated as `voters/numOfProcesses`.
* The last process, however, takes `voters/numOfProcesses` in addition to any remainder in case the number of `voters` is not divisible by `numOfProcesses`.
* All processes create an array called `votes`, which has the values from 1 to number of candidates.
* For every voter in the chunck, `votes` array is shuffled and then assigned to `sendbuf` which is a dynamic array of size chunk * number of candidates.
* `shuffle` function shuffles the order of values. This ensures that a single voter would never have repeated candidate numbers in his/her votes.
* To ensure that shuffle creates different values for each process, and different values everytime a user runs the program, we used `srand(time(0)+rank))`
* Then, to gather the results of all processes,`Gatherv` was used at root node.
* Since each process should be displaced for a size of chunk * number of candidates, displacement was calculated as `i*(voters/numOfProcesses)*cands` where i is the rank of the process.
* Recieve counts for all processes except the last was `(voters/numOfProcesses)` while the last process was that in addition to the remainder from voters/numOfProcesses, all multiplied by the number of candidates since each line countains not only 1 value, but number of candidates.
* The results are then serially written into a file.
## Task 2 result calculation
1. Read the file in parallel
    * All filed read the first two lines to know C and V
    * Every line is the same length , V whitespaces, and sum of digits from 1 ... C. Which is calculated using `getLineChars` 
    * Use rank to seek the file to  the section to read. 
    * Each  process processes `V/P` where P is the number of processes
    * When V is not divisible by P, the last process does the rest.
    * Every line is read using `readLine` function that reads the integers into an array
2. First round
    * The first vote from each line is used.
    * Then after all the votes for a process is done, the result is reduced to have the total votes at process 0
    * The votes are converted to percentages and printed to `stdout`
    * If any candidate gets more than 50% then we have a winner
    * Otherwise we move on to Round 2
    * In either of these scenarios, we broadcast the winner variable to all processes, to inform other processes if a winner was decided.
3. Second round
    * The first vote from each line is used if it is one of the 2 winning candidates from round 1, otherwise discard and read next from the same line.
    * Then after all the votes for a process is done, the result is reduced to have the total votes at process 0
    * The votes are converted to percentages and printed to `stdout`
    * Get the candidate with score >= 50% and print it as winner.
