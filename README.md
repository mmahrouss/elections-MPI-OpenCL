# elections-MPI
Distributed election program using MPI

# Team:
* Mohamed Kasem 201601144
* Mohamed Mahrous 201601078
* Alaa Roshdy 201600031
* Mohamed Mostafa 201600236

## Task 1 input generation
* Generate file in parallel
* I/P C (number of candidates), V voters
* Distribute the number of lines (V)
* for each node
    * seed:
        * staging:
            * rank
        * production:
            * time + rank
    * Generate `Array[V/P x C]`
    * Handle last process where the V is not multiple P
* `Gatherv` at root node
* Write file
## Task 2 result calculation
<!-- * Vote weight score = `C - i` -->
* Step 1. I/O
    * Read the file in parallel
    * read the first 2 lines
    * decide which lines to read from file
    * For each process:
        * calculate score for top candidate only
        * `Array[C]` 
        * Use `reduce` to get total score at root
    * At root check if first round settles it
    * If not
        * `broadcast` top 2 candidates for 2nd round
        * `Array[2]` result
        * `reduce` again
    * `stdout` result
