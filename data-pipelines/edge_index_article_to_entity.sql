SELECT art.article_node_id, et.entity_node_id
FROM {{ ref('geg_week_10_21') }} as geg
join {{ ref('entities_table') }} as et on (geg.mid = et.mid)
join {{ ref('articles_table') }} as art on (geg.url = art.url)
group by article_node_id, entity_node_id