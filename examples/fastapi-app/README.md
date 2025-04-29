# Example Fast API app instrumented with Laminar

## Installation

### 1. clone the repository

```
git clone https://github.com/lmnr-ai/lmnr-python
```

### 2. Open the directory

```
cd lmnr-python/examples/fastapi-app
```

### 3. Set up the environment variables

```
cp .env.example .env
```

And then fill in the `.env` file. Get [Laminar project API key](https://docs.lmnr.ai/tracing/introduction#2-initialize-laminar-in-your-application). Get [OpenAI API key](https://platform.openai.com/api-keys)

### 4. Install the dependencies

```
uv venv
```

```
source .venv/bin/activate
```

```
uv lock && uv sync
```

You may use `pip` or any other dependency manager of your choice instead of `uv`.

## Run the app

```
fastapi dev src/main.py --port 8011
```

## Test the call with curl

```
curl --location 'localhost:8011/api/v1/tickets/classify' \
--header 'Content-Type: application/json' \
--data-raw '{
    "title": "Can'\''t access my account",
    "description": "I'\''ve been trying to log in for the past hour but keep getting an error message",
    "customer_email": "user@example.com"
}'
```

## See the results on Laminar dashboard

In your browser, open https://www.lmnr.ai, navigate to your project's traces page, and you will see the auto-instrumented OpenAI span
