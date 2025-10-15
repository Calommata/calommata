from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello_fastapi() -> str:
    return "Hello World!"
