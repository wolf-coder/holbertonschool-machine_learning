-- Improve 34-log_stats.py by adding the top 10 of the most present IPs in the collection nginx of the database logs:
--   + The IPs top must be sorted (like the example below)
CREATE INDEX idx_name_first ON names (name(1))
