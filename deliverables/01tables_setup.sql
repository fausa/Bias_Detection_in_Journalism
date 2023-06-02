USE 509_final_proj;

DROP TABLE IF EXISTS nar_temp;
DROP TABLE IF EXISTS news_articles;

/**/
CREATE TABLE news_articles
(
text_id SMALLINT UNSIGNED AUTO_INCREMENT,
source_name VARCHAR(1000),
author VARCHAR(1000),
title VARCHAR(1000),
url VARCHAR(1000),
publish_date VARCHAR(30),
article_text LONGTEXT,
content LONGTEXT,
CONSTRAINT pk_text_id PRIMARY KEY (text_id)
);
DESC news_articles;

/**/
CREATE TABLE nar_temp
(
source_name VARCHAR(1000),
author VARCHAR(1000),
title VARCHAR(1000),
url VARCHAR(1000),
publish_date VARCHAR(30),
content LONGTEXT
);
DESC nar_temp;
