import asyncio
from apps.api.core.schema_bootstrap import bootstrap_db

print(asyncio.run(bootstrap_db('db/schema.sql')))
