from operator import itemgetter
with open("input_file.txt", "r") as fd:
    C, V = int(fd.readline()), int(fd.readline())
    votes_sum = {
        str(i): 0
        for i in range(1, C + 1)
    }
    for line in fd:
        votes_sum[line.split()[0]] += 1

print("Round 1 results\n============================")
for cand, votes in votes_sum.items():
    print(
        f"Candidate [{cand}] got {votes}/{V} which is {round(votes/V * 100, 2)}%")
    if votes / V > 0.5:
        print(cand, 1)
        break
else:
    # ROUND 2
    top2 = sorted(((cand, votes)
                   for cand, votes in votes_sum.items()), key=itemgetter(1), reverse=True)
    with open("input_file.txt", "r") as fd:
        next(fd)
        next(fd)
        round2_votes = {str(top2[0][0]): 0, str(top2[1][0]): 0}
        for line in fd:
            for cand in line.split():
                if cand in round2_votes:
                    round2_votes[cand] += 1
                    break

    print("Round 2 results\n============================")
    for cand, votes in round2_votes.items():
        print(
            f"Candidate [{cand}] got {votes}/{V} which is {round(votes/V * 100, 2)}%")
    winner = sorted(((cand, votes)
                     for cand, votes in round2_votes.items()), key=itemgetter(1), reverse=True)[0][0]
    print(winner, 2)
