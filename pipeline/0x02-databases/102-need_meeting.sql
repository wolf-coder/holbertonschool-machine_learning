-- Write a SQL script that creates a view need_meeting that lists all students that have a score under 80 (strict) and no last_meeting or more than 1 month.
--  + The view need_meeting should return all students name when:
--  + They score are under (strict) to 80
--  + AND no last_meeting date OR more than a month
CREATE VIEW need_meeting AS SELECT name FROM students
WHERE score < 80
      AND (last_meeting IS NULL OR DATEDIFF(CURDATE(),last_meeting) > 30);
