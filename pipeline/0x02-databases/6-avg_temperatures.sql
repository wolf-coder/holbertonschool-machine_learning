-- 6. Temperatures grouped by city
SELECT city, AVG(value) as avg_temp FROM temperatures GROUP BY CITY ORDER BY avg_temp DESC;
