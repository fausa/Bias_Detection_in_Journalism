USE 509_final_proj;

SELECT * FROM nar_temp;
SELECT * FROM news_articles;

SELECT
source_name,
COUNT(*) AS source_count,
COUNT(*) / sum(count(*)) over () AS source_dist
FROM news_articles
GROUP BY source_name;

SELECT DISTINCT source_name, title, article_text
FROM news_articles
WHERE article_text IS NOT NULL
	AND (source_name="CNN" OR source_name="The Washington Post" OR source_name="Fox News" OR source_name="Breitbart News")
LIMIT 2000;

SELECT
source_name,
COUNT(*) AS source_count,
COUNT(*) / sum(count(*)) over () AS source_dist
FROM news_articles
WHERE article_text IS NOT NULL
	AND (source_name="CNN" OR source_name="The Washington Post" OR source_name="Fox News" OR source_name="Breitbart News")
GROUP BY source_name;

SELECT source_name, title, article_text
FROM news_articles
WHERE article_text IS NULL
LIMIT 10000;

DESC news_articles;