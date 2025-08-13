import pytest
from sqlmodel import Session
from app.core.database import engine, SQLModel

@pytest.fixture(scope="session", autouse=True)
def create_test_db():
    SQLModel.metadata.create_all(engine)
    yield
    SQLModel.metadata.drop_all(engine)