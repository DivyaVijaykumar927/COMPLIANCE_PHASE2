# PGVectorScale Setup Guide

## Prerequisites
- Docker
- Python 3.7+
- OpenAI API key
- PostgreSQL GUI client (e.g., TablePlus, pgAdmin, or DBeaver)

## Steps

### 1. Create Dataset
Ensure you have your dataset ready inside a `dataset` folder.

### 2. Set Up Docker Environment
Create a `docker-compose.yml` file with the following content:

```yaml
version: '3'
services:
  timescaledb:
    image: timescale/timescaledb-ha:pg16
    container_name: timescaledb
    environment:
      - POSTGRES_DB=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  timescaledb_data:
```

Run the following command to start the container:
```sh
docker compose up -d
```

### 3. Verify Running Containers
Check if the container is running:
```sh
docker ps
```
If no containers are running, list all containers (including stopped ones):
```sh
docker ps -a
```
To start a stopped container:
```sh
docker start <container_id_or_name>
```
To check logs for troubleshooting:
```sh
docker logs <container_id_or_name>
```

### 4. Connect to the Database
Use a PostgreSQL client with the following credentials:
- **Host**: `localhost`
- **Port**: `5432`
- **Username**: `postgres`
- **Password**: `password`
- **Database**: `postgres`

### 5. Insert Vectors into Database
Use `insert_vectors.py` to generate embeddings using OpenAI’s `text-embedding-3-small` model and insert them into the database.
```sh
python insert_vectors.py
```

### 6. Perform Similarity Search
Use `similarity_search.py` to perform vector similarity searches.
```sh
python similarity_search.py
```

### 7. Optimize Performance with ANN Indexes
For large datasets, use the following ANN index options:
- `timescale_vector_index` (DiskANN-inspired graph index, recommended)
- `HNSW` (Hierarchical Navigable Small World graph index)
- `IVFFLAT` (Inverted file index)

Creating an index on the embedding column improves query performance:
```sql
CREATE INDEX ON vectors USING hnsw (embedding);
```

### 8. Restart Docker if Necessary
If containers fail to start, restart Docker:
```sh
sudo systemctl restart docker  # Linux/macOS
```
Or restart Docker Desktop on Windows/macOS.

### 9. Access and Manage Data
Access the database using a PostgreSQL GUI client to inspect stored vectors and run SQL queries.

---

This setup ensures an optimized PostgreSQL-based vector database with pgvectorscale for AI applications.

