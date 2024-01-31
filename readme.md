# Chatbot with PDF Upload Capability

This chatbot supports multiple PDF file uploads and offers multilingual chat capabilities. Before using it, please follow the instructions below to complete the necessary setup.

## Prerequisites

- Ensure you have the latest version of Python installed on your system.

## Setup Steps

### 1. Clone the repo.

Begin by cloning the `repo` to a suitable location on your computer.

### 2. Navigate to the Project Directory

Open your terminal and navigate to the project directory where you extracted the files. You can do this by using the `cd` command followed by the path to the directory.

### 3. Install Necessary Libraries

Depending on your operating system, use one of the following commands to install the required libraries:

- For Windows:
  ```shell
  pip install -r requirements.txt
  ```

- For macOS:
```shell
pip3 install -r requirements.txt
```

### 4. OpenAI API Key Configuration
Obtain your OpenAI API key. Once you have it, perform the following steps:

- Open the main.py file in a code or text editor.

- Locate the section with the following code:

```python
# ---------------------------------------->
##########################################
openai_api_key = "YOUR OPENAI API KEY"
##########################################
# ---------------------------------------->
```

- Replace "YOUR OPENAI API KEY" with your actual OpenAI API key.

### 5. Run the Web App
In the project directory, run the following command to start the web application:

```shell
uvicorn main:app
```

### 6. Accessing the Web App
Once set up, your web app should be accessible via your browser. You can interact with the chatbot and upload PDF files.