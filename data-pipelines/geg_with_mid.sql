{% call set_sql_header(config) %}
declare from_date timestamp default TIMESTAMP('{{ var("from_date") }}');
declare to_date timestamp default TIMESTAMP('{{ var("to_date") }}');
{% endcall %}

SELECT NET.HOST(geg.url) as host, geg.date, geg.url, geg.lang, geg.magnitude, geg.score, ent.name, ent.type, ent.mid, ent.wikipediaUrl, ent.numMentions, ent.avgSalience
FROM `gdelt-bq.gdeltv2.geg_gcnlapi` as geg, unnest(geg.entities) as ent
WHERE TIMESTAMP_TRUNC(geg.date, DAY) between from_date and to_date
and ent.mid != ''
