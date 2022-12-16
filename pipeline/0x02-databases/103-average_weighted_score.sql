-- Write a SQL script that creates a view need_meeting that lists all students that have a score under 80 (strict) and no last_meeting or more than 1 month.

-- Requirements:
-- 
--     The view need_meeting should return all students name when:
--         They score are under (strict) to 80
--         AND no last_meeting date OR more than a month

DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;
DELIMITER $$
CREATE PROCEDURE ComputeAverageWeightedScoreForUser (IN user_id INT)
BEGIN
    UPDATE users SET average_score=(
        SELECT SUM(weight*score)/SUM(weight) FROM corrections, projects
	WHERE corrections.user_id = user_id AND corrections.project_id=projects.id
    ) WHERE id=user_id;
END $$
DELIMITER ;
