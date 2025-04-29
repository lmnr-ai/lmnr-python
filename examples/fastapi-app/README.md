# Example Fast API app instrumented with Laminar

## Installation

### First, clone the repository

```
git clone https://github.com/lmnr-ai/lmnr-python
```

### Open the directory

```
cd lmnr-python/examples/fastapi-app
```

### Set up the environment variables

```
cp .env.example .env
```

And then fill in the `.env` file. Get [Laminar project API key](https://docs.lmnr.ai/tracing/introduction#2-initialize-laminar-in-your-application). Get [OpenAI API key](https://platform.openai.com/api-keys)

## Run the app

```
fastapi dev src/main.py
```