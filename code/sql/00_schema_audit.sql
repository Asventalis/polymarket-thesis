-- 00_schema_audit.sql: Audit schemas and row counts of universe views

SELECT 
    table_schema,
    table_name,
    table_type
FROM information_schema.tables 
WHERE table_schema IN ('main', 'universe')
ORDER BY table_schema, table_name;
