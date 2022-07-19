# Negative Engineering
---

**The negative engineering** is when engineer write the code to defensive code to make sure the positive code actully runs. For example what happens if data arrives malformed? What if the database goes down? What if the computer running the code fails? What if the code succeeds but the computer fails before it can report the success? Negative engineering is characterized by needing to anticipate this infinity of possible failures.

**Up to 90% of engineering time spent**

- Retries when APIs go down
- Malformed Data
- Notifications
- Observability into Failure
- Conditional Failure Login
- Timeouts

`Prefect` create the **Orchestration workflow** that aims to eliminate negative engineering. Reduce more time to 80% or 70% or more to help engineer can focus on the main task then the engieer can douple or triple developer productivity their work.

ref: [Prefect](https://www.prefect.io/)
ref: [Negative engineering](https://www.prefect.io/guide/blog/positive-and-negative-engineering/),or [more about topics](https://future.com/negative-engineering-and-the-art-of-failing-successfully/)
