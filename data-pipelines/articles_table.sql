-- create articles table
SELECT (ROW_NUMBER() OVER () - 1) AS article_node_id, host, url, max(lang) as lang, avg(magnitude) as magnitude, avg(score) as score
FROM {{ ref('geg_week_10_21') }}
group by host, url