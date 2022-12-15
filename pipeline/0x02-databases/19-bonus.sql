-- Write a SQL script that creates a stored procedure AddBonus that adds a new correction for a student.
--  + Procedure AddBonus is taking 3 inputs (in this order):
--  + user_id, a users.id value (you can assume user_id is linked to an existing users)
--  + project_name, a new or already exists projects - if no projects.name found in the table, you should create it
--  + score, the score value for the correction

DELIMITER $$
CREATE PROCEDURE AddBonus (
    IN user_id INT, project_name CHAR(255), score INT
)
BEGIN
    SET @count = (
        SELECT COUNT(*)
        FROM projects
        WHERE project_name = projects.name
    );
    IF @count = 0 THEN
        INSERT INTO projects (name) VALUES (project_name);
    END IF;
    SET @project = (
        SELECT id
        FROM projects
        WHERE project_name = projects.name
    );
    INSERT INTO corrections VALUES (user_id, @project, score);
END$$
DELIMITER ;
