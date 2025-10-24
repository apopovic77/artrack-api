#!/bin/bash
# Initialize Alembic for artrack-api

# Initialize Alembic
alembic init alembic

# Update alembic.ini to use environment variable for DB URL
sed -i.bak 's|sqlalchemy.url = .*|sqlalchemy.url = sqlite:////var/lib/api-arkturian/artrack.db|' alembic.ini

echo "✅ Alembic initialized"
echo "Next steps:"
echo "1. Edit alembic/env.py to import your models"
echo "2. Create first migration: alembic revision --autogenerate -m 'Initial migration'"
echo "3. Apply migration: alembic upgrade head"
