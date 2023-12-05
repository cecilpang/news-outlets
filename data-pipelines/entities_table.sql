select (ROW_NUMBER() OVER () - 1) AS entity_node_id, mid, type
FROM {{ ref('geg_week_10_21') }}
group by mid, type