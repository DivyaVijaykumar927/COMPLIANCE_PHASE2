# Use the TimescaleDB image as the base
FROM timescale/timescaledb-ha:pg16

# Switch to root user to ensure permission to install packages
USER root

# Install dependencies required to build PostgreSQL extensions
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    postgresql-server-dev-16

# Clone the pgvector repository and build the extension
RUN git clone https://github.com/pgvector/pgvector.git /tmp/pgvector && \
    cd /tmp/pgvector && \
    make && make install && \
    rm -rf /tmp/pgvector

# Switch back to the default PostgreSQL user for running the container
USER postgres

# Expose the PostgreSQL port
EXPOSE 5432
