-- Write a SQL script that creates an index idx_name_first_score on the table names and the first letter of name and the score.
--  + Import this table dump: names.sql.zip
--  + Only the first letter of name AND score must be indexed

CREATE INDEX idx_name_first_score ON names (name(1), score)
