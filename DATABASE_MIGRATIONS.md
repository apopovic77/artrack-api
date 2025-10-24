# Database Migration Guide

## Overview

The deployment pipeline now includes **automated database migrations with automatic backup and rollback** on failure.

## How It Works

When you run `./devops release`, the deployment will:

1. ✅ **Backup Database** - Creates timestamped backup before any changes
2. ✅ **Deploy Code** - Syncs new code to server
3. ✅ **Run Migrations** - Applies Alembic migrations automatically
4. ✅ **Restart Service** - Starts API with new code
5. ⚠️ **Auto-Rollback** - If migration fails, restores backup automatically

## Database Backup Location

All backups are saved to: `/var/backups/`

- **artrack-api**: `/var/backups/artrack-db-YYYYMMDD-HHMMSS.db`
- **storage-api**: `/var/backups/storage-db-YYYYMMDD-HHMMSS.db`

## Setting Up Migrations (First Time)

### 1. Initialize Alembic

```bash
cd /Volumes/DatenAP/Code/artrack-api

# Initialize Alembic
alembic init alembic

# Update alembic.ini to use environment variable
# Change: sqlalchemy.url = driver://user:pass@localhost/dbname
# To: sqlalchemy.url = sqlite:////var/lib/api-arkturian/artrack.db
```

### 2. Configure Alembic to Use Your Models

Edit `alembic/env.py`:

```python
# Add at the top
from artrack.models import Base
from artrack.database import SQLALCHEMY_DATABASE_URL
from sqlalchemy import engine_from_config, pool

# Update target_metadata
target_metadata = Base.metadata

# Update get_url() to use environment variable
def get_url():
    return SQLALCHEMY_DATABASE_URL
```

### 3. Create Initial Migration

```bash
# Generate migration from current models
alembic revision --autogenerate -m "Initial migration"

# Review the generated migration in alembic/versions/

# Test locally
alembic upgrade head
```

### 4. Commit and Deploy

```bash
git add alembic/
git commit -m "Add Alembic migrations"
./devops push "Add database migrations"
./devops release
# 🚀 Migrations will run automatically during deployment
```

## Making Schema Changes

### Example: Adding a New Column

1. **Update Model**:
```python
# artrack/models.py
class Track(Base):
    __tablename__ = "tracks"
    # ... existing fields ...
    difficulty_level = Column(String, nullable=True)  # NEW FIELD
```

2. **Generate Migration**:
```bash
alembic revision --autogenerate -m "Add difficulty_level to tracks"
```

3. **Review Migration**:
```bash
cat alembic/versions/xxx_add_difficulty_level.py
# Verify the upgrade() and downgrade() functions look correct
```

4. **Test Locally**:
```bash
# Apply migration
alembic upgrade head

# Test rollback
alembic downgrade -1

# Re-apply
alembic upgrade head
```

5. **Deploy**:
```bash
./devops push "Add difficulty level to tracks"
./devops release
```

## Deployment Flow with Migrations

```
./devops release
    ↓
Merge dev → main
    ↓
Push to GitHub
    ↓
GitHub Actions Triggered
    ↓
📦 Backup Database (/var/backups/artrack-db-20251022-143000.db)
    ↓
🚀 Deploy Code (rsync)
    ↓
📦 Install Dependencies
    ↓
🔄 Run Alembic Migrations (alembic upgrade head)
    ↓
    ├─ ✅ Success → Restart Service
    │
    └─ ❌ Failure → Restore Backup & Exit
```

## Multiple Deployment Targets

When deploying to multiple servers (e.g., staging + production):

### Option 1: Same Workflow, Different Secrets

Create separate GitHub Actions secrets for each environment:

```yaml
# .github/workflows/deploy-staging.yml
on:
  push:
    branches: [staging]
env:
  DEPLOY_HOST: ${{ secrets.STAGING_DEPLOY_HOST }}

# .github/workflows/deploy-production.yml
on:
  push:
    branches: [main]
env:
  DEPLOY_HOST: ${{ secrets.PROD_DEPLOY_HOST }}
```

### Option 2: Manual Selection

```bash
# Deploy to staging
gh workflow run deploy.yml --ref main -f environment=staging

# Deploy to production
gh workflow run deploy.yml --ref main -f environment=production
```

## Migration Safety Checklist

Before releasing schema changes:

- [ ] Migration generated with `alembic revision --autogenerate`
- [ ] Migration reviewed manually
- [ ] Migration tested locally (`alembic upgrade head`)
- [ ] Rollback tested (`alembic downgrade -1`)
- [ ] Backward compatibility considered (can old code run on new schema?)
- [ ] Data migration scripts added if needed
- [ ] Committed to dev branch
- [ ] CI tests pass on dev branch

## Troubleshooting

### Migration Failed on Server

1. **Check Logs**:
```bash
ssh root@arkturian.com "tail -100 /var/log/artrack-api-error.log"
```

2. **Check Backup**:
```bash
ssh root@arkturian.com "ls -lh /var/backups/artrack-db-*"
```

3. **Manual Rollback** (if auto-rollback failed):
```bash
ssh root@arkturian.com
cp /var/backups/artrack-db-LATEST.db /var/lib/api-arkturian/artrack.db
systemctl restart artrack-api
```

### View Migration History

```bash
# On server
cd /var/www/api-artrack.arkturian.com
alembic current
alembic history
```

### Manually Run Migration

```bash
ssh root@arkturian.com
cd /var/www/api-artrack.arkturian.com
/root/.pyenv/versions/3.11.9/bin/alembic upgrade head
```

## Best Practices

1. **Always test migrations locally first**
2. **Keep migrations small and focused**
3. **Add data migrations separately from schema changes**
4. **Never modify old migrations** - create new ones
5. **Document complex migrations** with comments
6. **Consider backward compatibility** - can old code run on new schema during deployment?

## Example: Complex Migration with Data Transform

```python
# alembic/versions/xxx_migrate_status_enum.py
def upgrade():
    # 1. Add new column
    op.add_column('tracks', sa.Column('status_new', sa.String(), nullable=True))

    # 2. Migrate data
    connection = op.get_bind()
    connection.execute("""
        UPDATE tracks
        SET status_new = CASE
            WHEN status = 0 THEN 'draft'
            WHEN status = 1 THEN 'published'
            WHEN status = 2 THEN 'archived'
        END
    """)

    # 3. Drop old column and rename new
    op.drop_column('tracks', 'status')
    op.alter_column('tracks', 'status_new', new_column_name='status')

def downgrade():
    # Reverse the migration
    op.add_column('tracks', sa.Column('status', sa.Integer(), nullable=True))
    # ... reverse data migration ...
    op.drop_column('tracks', 'status_new')
```

## Summary

✅ **Automated** - Migrations run automatically on deployment
✅ **Safe** - Database backed up before every migration
✅ **Resilient** - Auto-rollback on failure
✅ **Auditable** - All backups timestamped and saved
✅ **Repeatable** - Same process for all environments

Your database schema changes are now fully integrated into the CI/CD pipeline!
