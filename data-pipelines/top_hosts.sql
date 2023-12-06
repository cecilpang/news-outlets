SELECT h.*, t50.host as top_site
FROM  `gannett-datarevenue.zz_test_pang.top_50_news_sites` as t50
join {{ ref('hosts_table') }} as h
on regexp_contains(h.host, t50.host) and net.reg_domain(t50.host) = net.reg_domain(h.host)