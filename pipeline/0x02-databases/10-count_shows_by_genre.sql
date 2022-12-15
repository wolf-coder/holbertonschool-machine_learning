-- 10. Number of shows by genre
SELECT tv_g.name AS genre, COUNT(*) AS number_of_shows FROM tv_genres tv_g, tv_show_genres tv_s_g, tv_shows tv_s
WHERE tv_g.id=tv_s_g.genre_id AND tv_s.id=tv_s_g.show_id
GROUP BY tv_g.name HAVING COUNT(*) != 0 ORDER BY number_of_shows DESC;
