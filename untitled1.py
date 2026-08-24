import random

class NQueensCSP:
    def __init__(self, N):
        self.N = N
        self.domains = list(range(N))

    def conflicts(self, assignment):
        """Returns the number of conflicts in the current assignment."""
        count = 0

        for i in range(self.N):
            for j in range(i + 1, self.N):
                if (assignment[i] == assignment[j] or
                    abs(assignment[i] - assignment[j]) == j - i):
                    count += 1

        return count

    def min_conflicts(self, max_steps=1000):
        """Min-conflicts algorithm to solve the N-Queens problem."""

        assignment = [
            random.choice(self.domains)
            for _ in range(self.N)
        ]

        for _ in range(max_steps):
            if self.conflicts(assignment) == 0:
                return assignment

            conflicted_vars = [
                i for i in range(self.N)
                if any(
                    assignment[i] == assignment[j] or
                    abs(assignment[i] - assignment[j]) == abs(i - j)
                    for j in range(self.N)
                    if i != j
                )
            ]

            var = random.choice(conflicted_vars)

            min_conflict_value = min(
                self.domains,
                key=lambda val: self.conflicts(
                    assignment[:var] + [val] + assignment[var + 1:]
                )
            )

            assignment[var] = min_conflict_value

        return None


N = 8
nqueens = NQueensCSP(N)

solution = nqueens.min_conflicts()

if solution:
    print("Found:", solution)
else:
    print("Nothing found")