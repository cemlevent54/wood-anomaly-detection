# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.app.routes.test_controller import router as test_router

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # all domains
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, PUT, DELETE etc.
    allow_headers=["*"],  # Authorization, Content-Type etc.
)

app.include_router(test_router, prefix="/test", tags=["test"])

@app.get("/")
def read_root():
    return {"message": "FastAPI integrated with CORS!"}
