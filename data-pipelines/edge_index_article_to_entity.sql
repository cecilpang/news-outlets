SELECT art.article_node_id, et.entity_node_id
FROM `gannett-datarevenue.zz_test_pang.geg_week_10_21` as geg
join `gannett-datarevenue.zz_test_pang.entities_table` as et on (geg.mid = et.mid)
join `gannett-datarevenue.zz_test_pang.articles_table` as art on (geg.url = art.url)
group by article_node_id, entity_node_id