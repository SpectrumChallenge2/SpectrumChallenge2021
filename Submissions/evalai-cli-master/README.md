# EvalAI-CLI

<b>Official Command Line utility to use EvalAI in your terminal.</b>

EvalAI-CLI is designed to extend the functionality of the EvalAI web application to command line to make the platform more accessible and terminal-friendly to its users.

------------------------------------------------------------------------------------------

[![Join the chat at https://gitter.im/Cloud-CV/EvalAI](https://badges.gitter.im/Cloud-CV/EvalAI.svg)](https://gitter.im/Cloud-CV/EvalAI?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/Cloud-CV/evalai-cli.svg?branch=master)](https://travis-ci.org/Cloud-CV/evalai-cli)
[![Coverage Status](https://coveralls.io/repos/github/Cloud-CV/evalai-cli/badge.svg?branch=master)](https://coveralls.io/github/Cloud-CV/evalai-cli?branch=master)
[![Documentation Status](https://readthedocs.org/projects/markdown-guide/badge/?version=latest)](https://cli.eval.ai/)

## Installation

EvalAI-CLI and its required dependencies can be installed using pip:
```sh
pip install evalai
```
Once EvalAI-CLI is installed, check out the [usage documentation](https://cli.eval.ai/).

## Contributing Guidelines

If you are interested in contributing to EvalAI-CLI, follow our [contribution guidelines](https://github.com/Cloud-CV/evalai-cli/blob/master/.github/CONTRIBUTING.md).

## Development Setup

1. Setup the development environment for EvalAI and make sure that it is running perfectly.

2. Clone the evalai-cli repository to your machine via git

    ```bash
    git clone https://github.com/Cloud-CV/evalai-cli.git evalai-cli
    ```

3. Create a virtual environment

    ```bash
    cd evalai-cli
    virtualenv -p python3 venv
    source venv/bin/activate
    ```

4. Install the package locally

    ```bash
    pip install -e .
    ```
 
5. Change the evalai-cli host to make request to local EvalAI server running on `http://localhost:8000` by running:
   
   ```bash
   evalai host -sh http://localhost:8000
   ```


6. Login to cli using the command ``` evalai login```
Two users will be created by default which are listed below -

    ```bash
    Host User - username: host, password: password
    Participant User - username: participant, password: password
    ```
