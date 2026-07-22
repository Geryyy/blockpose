## Implementation workflow

For non-trivial changes:

1. Use the Plan agent before implementation.
2. Delegate implementation to a general-purpose worker agent.
3. Run the verification agent after implementation.
4. The verifier must inspect the diff, tests, correctness, regressions,
   edge cases, error handling and security.
5. Fix critical and high-severity findings before considering the task complete.
6. Run verification again after substantial fixes.

## Filesystem safety

- Never use `git clean` or `git reset --hard`.