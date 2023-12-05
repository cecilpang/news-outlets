from dbt.cli.main import dbtRunner, dbtRunnerResult
import json


def main():
    # initialize
    dbt = dbtRunner()

    # create CLI args as a list of strings
    # e.g. cli_args = ["run", "--select", "tag:my_tag"]
    run_args = {}
    #cli_args = ["run", "--select", "geg_week_10_21", "--vars", json.dumps(run_args)]
    cli_args = ["run", "--select", "edge_index_article_to_entity", "--vars", json.dumps(run_args)]

    # run the command
    res: dbtRunnerResult = dbt.invoke(cli_args)

    # inspect the results
    for r in res.result:
        print(f"{r.node.name}: {r.status}")


if __name__ == "__main__":
    main()