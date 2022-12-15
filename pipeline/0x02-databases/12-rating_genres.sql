-- Import the database dump from hbtn_0d_tvshows_rate to your MySQL server: download
-- + Each record should display: tv_genres.name - rating sum
-- + Results must be sorted in descending order by their rating
-- + You can use only one SELECT statement
-- + The database name will be passed as an argument of the mysql command

SELECT tvg.name, SUM(rate) AS rating FROM tv_genres tvg, tv_show_genres tvsg, tv_show_ratings tvsr
WHERE tvg.id=tvsg.genre_id AND tvsg.show_id=tvsr.show_id
GROUP BY tvg.name ORDER BY SUM(rate) DESC;
