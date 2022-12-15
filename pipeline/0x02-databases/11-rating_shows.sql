-- Import the database hbtn_0d_tvshows_rate dump to your MySQL server: download
--  + Each record should display: tv_shows.title - rating sum
--  + Results must be sorted in descending order by the rating
--  + You can use only one SELECT statement

SELECT tvs.title, SUM(rate) AS rating FROM tv_shows tvs, tv_show_ratings tvsr
WHERE tvs.id=tvsr.show_id
GROUP BY tvs.id ORDER BY SUM(rate) DESC;
