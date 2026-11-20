-- Files attached to a user message: [{ id, name, mime, size }]. Bytes live in the
-- bucket at console/files/<chat_id>/<id> (see fileKey in src/agent.ts).
alter table messages add column attachments jsonb not null default '[]';
