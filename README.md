# `Table of Contents`

- [Introduction](#introduction)
- [Build and Run](#build-and-run)
  - [Prerequisites](#prerequisites)
  - [Github Codespace](#github-codespace)
  - [Local Dev Container](#local-dev-container)
    - [Installation](#installation)
    - [Running Services Locally](#running-services-locally)
  - [Running Evaluations](#running-evaluations)
  - [Notes](#notes)
    - [Switching to Visual Studio Code From Browser](#switching-to-visual-studio-code-from-browser)
    - [Rebuilding the Container](#rebuilding-the-container)

## Introduction

There are two main branches on this repository. The `main` branch contains the following:

   ![main-ui](./docs/images/main-ui.png)

   1. The source code for the Streaming enabled ordering assistant Chatbot. The chatbot is a conversational AI that can be used to place orders for food and drinks. It allows users to select different models for testing as well as tone instructions for the LLM interactions all from drop down lists.

   The chatbot is built using the AOAI, PromptFlow and the FastAPI framework. The frontend is built using Streamlit. This code is located in `streaming_ordering_chatbot` directory. The flows are built utilizing the flex flows (latest version of PromptFlow) and the code is located in the `flows` directory. This branch contains order item validation specific to the example Contos burger menu.

   2. Example flows built using older promptflow implementation and sample code for evaluation of the flows. The code is located in the `demo` directory. For instruction on how to run the evaluations, see the [Running Evaluations](#running-evaluations) section.

      [High Level Architecture](./docs/architecture.md)

      [How the Chatbot Works](./docs/streaming_with_output_validation.md)

      [Overview of Observability](./docs/observability.md)

The second main branch is called [main-v2-flexible-menus](https://github.com/cse-labs/streaming-ordering-chatbot/tree/main-v2-flexible-menus). This branch contains the same features as above but it is more flexible by allowing the user to select different menu from a dropdown and can be substituted with any sample menu following the same pattern. It currently has a drop down to select one of the 3 menus (Burger(default), Coffee and Taco). The code defaults to burger menu. For the purpose of flexibility to demo the concepts, item validations have been removed to adhere to a more general schema.

   ![main-v2-ui](./docs/images/main-v2-ui.png)

## Build and Run

This section provides step-by-step instructions on what is needed to build and run this solution. There are two option for running the solution:

- Github Codespace
- Local Dev Container

### Prerequisites

   ``**You can skip this step if you are using the Github Codespace container.**``

Before proceeding, ensure that you have the following software dependencies installed:

- [VSCODE](https://code.visualstudio.com/download)
- [Docker](https://www.docker.com/products/docker-desktop/)

### Github Codespace

Github codespace enables you to run the code in browser or in Visual Studio Code (if already installed) without having to install any dependencies on your local machine. To run the code in Github Codespace, follow these steps:

1. To build your codespace click on the `Code` button on the top right of the repository and select `Codespaces` and then `Create codespace on main` (building off of the main branch)

   ![github](./docs/images/navigateFromGithub.png)

   This will open the your browser and starts setting up the environment and building the container. This may take a few minutes and you will see the progress in the bottom left corner of the screen. Should also wait for all extensions to be installed as well.

2. To run the services click on the `debug extension` on the left side of the screen and select the `Run FastAPI and React` configuration from the drop down menu at the top of the screen.

   ![debugRun](./docs/images/browserDebug.png)

3. This will run both the service on the backend and frontend ports. You can check the progress on both services in the terminals opened automatically.

   - The Front end service will wait to start until the backend service start up is complete, this is achieved by adding a task to the task.json under `.vscode` folder.

   You can access the frontend by clicking on the URL shown in the frontend app's terminal while pressing `Ctrl` and selecting `Open` button in the pop up window.

   ![openBrowser](./docs/images/serverUrl.png)

### Local Dev Container

#### Installation

To get started, follow these steps:

1. Clone the [repository](https://dev.azure.com/hermescrew/Hermes%20Crew/_git/streaming-ordering-chatbot).
2. Open the project in Visual Studio Code.
3. Open the project folder in container mode. This will build the container and install the necessary dependencies. You can achieve this by pressing CTRL+SHIFT+P and typing `Remote-Containers: Open Folder in Container`.
4. After the container is built, navigate to the `deployment/script` directory open the `azure-resource-deployment.sh` file in the terminal. It has default values for required parameters such as location, resource group, and account names.
5. run the `deploy_azure_resources.sh` script in the terminal.

   ```bash
   ./azure-resource-deployment.sh
   ```

   - The script will run the `az login` command to authenticate with your Azure account.
   - It will list the available subscriptions, select the subscription you want to use (type in the row number and hit enter).
   - This script will deploy the necessary resources to Azure and print out the required values for the `.env` file to run the code in a vscode container.
6. Create a new file named `.env` in the root directory copying the `.env.example` file.
   - Fill in the necessary environment variables in the `.env` file from the output of the deployment script.

   - For more details on the deployment script, see the [Azure Deployment Script](./deployment/script/readme.md) documentation.

#### Running Services Locally

To run the container in debug mode, follow these steps:

1. To run the services locally in debug mode you can click on the debug extension on the left side of the screen and select the "Run FastAPI and React" configuration from the drop down menu at the top of the screen.

   ![debugRun](./docs/images/debugRun.png)

2. This will run both the service on the backend and frontend ports. You can access the frontend at `http://localhost:8051` and the backend at `http://localhost:8000`.

### Running Evaluations

To run the evaluations, follow these steps:

- On your terminal, navigate to the `pf_utils` directory:

   ```bash
   cd demo/pf_utils/
   ```

- Run the following command to install the necessary dependencies:

   ```bash
   pip install -e .
   ```

- You would need to create a Pf connection to the Azure OpenAI service. You can do this by utilizing the PromptFlow extension that is preinstalled in the container.

   ![openBrowser](./docs/images/createConnection.png)

- Follow the instructions to create a connection to the Azure OpenAI service. Ensure that the connection name for all required flow.dag.yaml files is matching your actual connection.

   Currently the connection name is set to `connection: open_ai_connection`.

   ![openBrowser](./docs/images/connectionDetails.png)

- Navigate to the `execute_eval.py` file under `demo` directory.

   ![openBrowser](./docs/images/eval.png)

- Run the evaluations utilizing the debug configuration.
  
   ![openBrowser](./docs/images/evalRun.png)

### Notes

#### Switching to Visual Studio Code From Browser

If you decide to switch the codespace to run in Visual Studio Code from the browser, open the command window by pressing `Ctrl +Shift + P` and type `Open in VS Code Desktop`.

   ![navigate](./docs/images/vsCodeDesktop.png)

#### Rebuilding the Container

If you need to rebuild the container, you can do so by pressing `Ctrl +Shift + P` and typing `Rebuild Container`.

   ![rebuild](./docs/images/rebuildContainer.png)
