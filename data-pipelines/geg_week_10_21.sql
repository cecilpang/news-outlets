-- a week of geg data from Oct 15 to 21 inclusive
{% call set_sql_header(config) %}
declare from_date timestamp default TIMESTAMP('2023-10-15');
declare to_date timestamp default TIMESTAMP('2023-10-21');
{% endcall %}

SELECT NET.HOST(geg.url) as host, geg.date, geg.url, geg.lang, geg.magnitude, geg.score, ent.name, ent.type, ent.mid, ent.wikipediaUrl, ent.numMentions, ent.avgSalience
FROM `gdelt-bq.gdeltv2.geg_gcnlapi` as geg, unnest(geg.entities) as ent
WHERE TIMESTAMP_TRUNC(geg.date, DAY) between from_date and to_date
and ent.mid != ''