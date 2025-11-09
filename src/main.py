from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from views.api_router import router
import uvicorn

App = FastAPI(title="Sample FastAPI App", version="1.0.0")

App.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

App.include_router(router)
if __name__ == "__main__":
    uvicorn.run(App, host="0.0.0.0", port=8000,)
