from sqlmodel import create_engine, SQLModel

# These credentials must match your docker-compose.yml
DATABASE_URL = "postgresql://admin:password123@localhost:5432/fake_job_db"

engine = create_engine(DATABASE_URL)

def init_db():
    # This command builds the tables you defined in schema.py
    import backend.schema 
    SQLModel.metadata.create_all(engine)