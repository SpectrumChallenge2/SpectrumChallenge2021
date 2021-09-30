import json
import os

import click
from click import echo, style

from evalai.utils.config import AUTH_TOKEN_DIR, AUTH_TOKEN_PATH


@click.group(invoke_without_command=True)
@click.argument("auth_token")
def set_token(auth_token):
    """
    Configure EvalAI Token.
    """
    """
    Invoked by `evalai set_token <your_evalai_auth_token>`.
    """
    if not os.path.exists(AUTH_TOKEN_DIR):
        os.makedirs(AUTH_TOKEN_DIR)
    with open(AUTH_TOKEN_PATH, "w+") as fw:
        try:
            auth_token = {"token": "{}".format(auth_token)}  # noqa
            auth_token = json.dumps(auth_token)
            fw.write(auth_token)
        except (OSError, IOError) as e:
            echo(e)
        echo(
            style(
                "Success: Authentication token is successfully set.",
                bold=True,
                fg="green",
            )
        )
