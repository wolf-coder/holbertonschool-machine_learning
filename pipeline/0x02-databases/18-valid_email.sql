-- an SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed.

DELIMITER $$
CREATE TRIGGER email_check BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
	IF STRCMP(OLD.email, NEW.email) <> 0 THEN
	   SET NEW.valid_email = 0;
	END IF;	
END$$
DELIMITER ;
