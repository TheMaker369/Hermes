from fastapi import FastAPI

app = FastAPI()


@app.get("/status")
async def status():
    return {"status": "Hermes is online"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
