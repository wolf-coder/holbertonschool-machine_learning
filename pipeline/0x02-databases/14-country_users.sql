-- Write a SQL script that creates a table users following these requirements:
--  + With these attributes:
--  + id, integer, never null, auto increment and primary key
--  + email, string (255 characters), never null and unique
--  + name, string (255 characters)
--  + country, enumeration of countries: US, CO and TN, never null (= default will be the first element of the enumeration, here US)
--  + If the table already exists, your script should not fail
--  + Your script can be executed on any database

CREATE TABLE IF NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(256) NOT NULL UNIQUE,
    name VARCHAR(256),
    country ENUM('US', 'CO', 'TN') DEFAULT 'US',
    PRIMARY KEY (id)
);
