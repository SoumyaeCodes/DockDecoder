# ML_pipeline
A basic end-to-end ML-based pipeline project.

## Setting Up Your Development Environment
For an efficient development experience, please follow these two essential steps:
### 1. Install the Virtual Environment
First, create an isolated environment to house your project dependencies. Execute the following commands:

```bash
# For Windows
python -m venv venv
# For macOS and Linux
python3 -m venv venv
```

### 2. Install All Required Libraries
Next, activate your virtual environment and install the necessary libraries in one swift command:

```bash
# For Windows
.\venv\Scripts\activate
# For macOS and Linux
source venv/bin/activate
# Install all dependencies
pip install -r requirements.txt
```

### 3. 
To securely store your API keys and other sensitive information, you should create a `.env` file. Follow these steps:

```bash
# In the root directory of your project, create a new file named `.env`:
touch .env
```
```bash
#Open the `.env` file in your favorite text editor and add your API keys in the following format:
GROQ_API_KEY=<your-api-key-here>
```


With these steps, you’re well-equipped to embark on your coding journey!
