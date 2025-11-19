from fastapi import FastAPI

app = FastAPI(
    title="My FastAPI Service",
    description="This is an example API with Swagger documentation",
    version="1.0.0",
)

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI!"}

@app.get("/status", tags=["status"])
def status_check():
    return {"status": "ok"}