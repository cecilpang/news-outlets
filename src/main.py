from dbt.cli.main import dbtRunner, dbtRunnerResult
import json


def main():
    # initialize
    dbt = dbtRunner()

    # create CLI args as a list of strings
    # e.g. cli_args = ["run", "--select", "tag:my_tag"]
    run_args = {'from_date': '2023-09-01', 'to_date': '2023-11-30'}
    cli_args = ["run", "--select", "geg_with_mid", "--vars", json.dumps(run_args)]

    # run the command
    res: dbtRunnerResult = dbt.invoke(cli_args)

    # inspect the results
    for r in res.result:
        print(f"{r.node.name}: {r.status}")


if __name__ == "__main__":
    main()