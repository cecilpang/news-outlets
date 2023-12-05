SELECT ht.host_node_id, art.article_node_id
FROM {{ ref('geg_week_10_21') }} as geg
join {{ ref('hosts_table') }} as ht on (geg.host = ht.host)
join {{ ref('articles_table') }} as art on (geg.url = art.url)
group by host_node_id, article_node_id