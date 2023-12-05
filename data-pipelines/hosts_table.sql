SELECT (ROW_NUMBER() OVER ()) - 1 AS host_node_id, host
FROM {{ ref('geg_week_10_21') }}
group by host