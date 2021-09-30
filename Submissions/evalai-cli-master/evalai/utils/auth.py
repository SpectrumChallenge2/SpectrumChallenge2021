import os
import json
import sys
import requests

from click import echo, style
from evalai.utils.config import (
    AUTH_TOKEN_PATH,
    API_HOST_URL,
    EVALAI_ERROR_CODES,
    HOST_URL_FILE_PATH,
)
from evalai.utils.urls import URLS


requests.packages.urllib3.disable_warnings()


def get_user_access_token(token):
    """
    Returns user access token after login.
    """
    url = "{}{}".format(get_host_url(), URLS.get_access_token.value)
    try:
        headers = {"Authorization": "Token {}".format(token["token"])}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        if response.status_code in EVALAI_ERROR_CODES:
            echo(
                style(
                    "\nUnable to fetch auth token.\n",
                    bold=True,
                    fg="red",
                )
            )
        sys.exit(1)
    except requests.exceptions.RequestException:
        echo(
            style(
                "\nCould not establish a connection to EvalAI."
                " Please check the Host URL.\n",
                bold=True,
                fg="red",
            )
        )
        sys.exit(1)
    token = response.json()
    return token


def get_user_auth_token_by_login(username, password):
    """
    Returns user auth token by login.
    """
    url = "{}{}".format(get_host_url(), URLS.login.value)
    try:
        payload = {"username": username, "password": password}
        response = requests.post(url, data=payload)
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        if response.status_code in EVALAI_ERROR_CODES:
            echo(
                style(
                    "\nUnable to log in with provided credentials.\n",
                    bold=True,
                    fg="red",
                )
            )
        sys.exit(1)
    except requests.exceptions.RequestException:
        echo(
            style(
                "\nCould not establish a connection to EvalAI."
                " Please check the Host URL.\n",
                bold=True,
                fg="red",
            )
        )
        sys.exit(1)

    token = response.json()
    # Replace expiring auth token with access token
    token = get_user_access_token(token)
    return token


def get_user_auth_token():
    """
    Loads token to be used for sending requests.
    """
    if os.path.exists(AUTH_TOKEN_PATH):
        with open(str(AUTH_TOKEN_PATH), "r") as TokenObj:
            try:
                data = TokenObj.read()
            except (OSError, IOError) as e:
                echo(style(e, bold=True, fg="red"))
        data = json.loads(data)
        token = data["token"]
        return token
    else:
        echo(
            style(
                "\nThe authentication token json file doesn't exists at the required path. "
                "Please download the file from the Profile section of the EvalAI webapp and "
                "place it at ~/.evalai/token.json\n",
                bold=True,
                fg="red",
            )
        )
        sys.exit(1)


def get_request_header():
    """
    Returns user auth token formatted in header for sending requests.
    """
    header = {"Authorization": "Bearer {}".format(get_user_auth_token())}

    return header


def get_host_url():
    """
    Returns the host url.
    """
    if not os.path.exists(HOST_URL_FILE_PATH):
        return API_HOST_URL
    else:
        with open(HOST_URL_FILE_PATH, "r") as fr:
            try:
                data = fr.read()
                return str(data)
            except (OSError, IOError) as e:
                echo(style(e, bold=True, fg="red"))
