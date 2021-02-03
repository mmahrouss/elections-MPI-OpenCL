__kernel void
reduce(int C,                   // number of candidates
       __local int *localVotes, // Array of length C x n, n is number of work
                                // items per work group
       __global int
           *sumVotesOut // Array of length C x N, N is number of work groups
) {

  int num_wrk_items = get_local_size(0); // number of work items per work group
  int local_id = get_local_id(0);        // The index of the element (counted in
                                  // context of the local work group only)
  int group_id = get_group_id(0); // Gets the work group number
  // Loop for computing localVotes : divide WorkGroup into 2 parts
  // it works in the divide and conquer way
  // For a single work group, we loop over half the work items and sum with a
  // stride of half workgroup size
  for (uint stride = num_wrk_items / 2; stride > 0; stride /= 2) {
    // Waiting for each 2x2 addition into given workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    // Add elements 2 by 2 between local_id * C and local_id * C + stride * C
    if (local_id < stride) {
      for (uint candidate = 0; candidate < C; candidate += 1) {
        localVotes[local_id * C + candidate] +=
            localVotes[local_id * C + candidate + stride * C];
      }
    }
  }

  // Write result into sumVotesOut
  if (local_id == 0) {
    for (uint candidate = 0; candidate < C; candidate += 1) {
      sumVotesOut[group_id * C + candidate] = localVotes[candidate];
    }
  }
}

__kernel void
getVotes(int C,                    // number of candidates
         __global int *firstVotes, // Array of length N (number of voters +
                                   // padding to work group size)
         __local int *localVotes,  // Array of length C x n, n is number of work
                                   // itemsper workgroup
         __global int
             *sumVotesOut // Array of length C x N, N is number of work groups
) {
  int num_wrk_items = get_local_size(0); // n
  int local_id = get_local_id(0);
  int group_id = get_group_id(0);

  // Get local vote
  // firstVotes[group_id][local_id]

  int vote = firstVotes[group_id * num_wrk_items + local_id];

  // init localVotes to all zeros
  for (uint candidate = 0; candidate < C; candidate += 1) {
    localVotes[local_id * C + candidate] = 0;
  }

  // localVotes[local_id][vote] = 1
  if (vote > -1) {
    localVotes[local_id * C + (vote - 1)] = 1;
  }
  reduce(C, localVotes, sumVotesOut);
}

__kernel void iterativeReducer(
    int C,     // number of candidates
    int nPrev, // previoud number of groups
    __global int *sumVotesIn,
    __local int *localVotes, // Array of length C x n, n is number of work items
                             // per work group
    __global int
        *sumVotesOut // Array of length C x N, N is number of work groups
) {
  int local_id = get_local_id(0);
  int global_id = get_global_id(0);
  for (uint candidate = 0; candidate < C; candidate += 1) {
    localVotes[local_id * C + candidate] =
        (global_id >= nPrev) ? 0 : sumVotesIn[global_id * C + candidate];
  }
  reduce(C, localVotes, sumVotesOut);
}